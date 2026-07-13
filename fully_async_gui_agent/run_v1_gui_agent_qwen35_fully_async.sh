#!/usr/bin/env bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# GUI Agent (Computer-Use Agent) PPO training on the **V1 trainer**,
# `fully_async` (streaming) mode — TransferQueue-native.
#
# This is the streaming sibling of `run_v1_gui_agent_separate_async.sh`: same
# agent loop, tools, data, model, desktop-env and rollout-correction settings;
# the ONLY difference is how prompts are produced.
#
#   separate_async : step() feeds exactly one batch per training step
#                    (+ num_warmup_batches up front) — production is locked to
#                    consumption.
#   fully_async    : an autonomous background StreamingFeeder thread continuously
#                    streams prompts into TransferQueue, bounded by a staleness /
#                    in-flight budget
#                      max_inflight = (1 + staleness_threshold)
#                                     * parameter_sync_step * train_batch_size
#                    while step() only samples + trains. Generation overlaps
#                    training; the feeder is paused around each weight sync.
#
# Off-policy handling: streaming BOUNDS staleness via the feeder budget and
# CORRECTS residual off-policyness via rollout correction (bypass mode + RS/TIS),
# rather than hard-dropping in the replay buffer (which can empty a batch). So the
# trainer-side staleness gate is OFF by default (max_off_policy_strategy=none),
# unlike separate_async which defaults to `drop`.
#
# Config mapping vs run_v1_gui_agent_separate_async.sh:
#   trainer.v1.trainer_mode                       separate_async -> fully_async
#   trainer.v1.separate_async.num_warmup_batches  -> trainer.v1.fully_async.num_warmup_batches (0)
#   trainer.v1.separate_async.parameter_sync_step -> trainer.v1.fully_async.parameter_sync_step
#   (new)                                         -> trainer.v1.fully_async.staleness_threshold
#   (new)                                         -> trainer.v1.fully_async.feeder_poll_interval
#   trainer.v1.sampler.max_off_policy_strategy     drop -> none
#
# Prerequisites (unchanged):
#   1. A running desktop-env service (DESKTOP_API_BASE_URL).
#   2. A Qwen3.5 VLM checkpoint (default: /efs/data/models/Qwen3.5-9B).
#   3. A parquet dataset with prompt / extra_info.task_id / extra_info.question.
#   4. A verl-v1 checkout (V1 trainer with the `fully_async` mode) plus
#      `transfer_queue` (TransferQueue) on EVERY Ray node, and this recipe
#      importable as `recipe.fully_async_gui_agent` (use the v1-gui-agent branch).

set -xeuo pipefail
export HYDRA_FULL_ERROR=1
export HF_HOME="/efs/data/hf"

# ================= process resource limits =================
ulimit -n 1048576 || true
ulimit -c unlimited || true

export WANDB_API_KEY=${WANDB_API_KEY:-}
export RAY_USE_UVLOOP=${RAY_USE_UVLOOP:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF:-expandable_segments:True}}

# ================= paths =================
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${VERL_ROOT:-}" ]]; then
    if [[ -d "verl/trainer/config" ]]; then
        VERL_ROOT="$(pwd)"
    elif [[ -d "${RECIPE_DIR}/../../verl-v1/verl/trainer/config" ]]; then
        VERL_ROOT="$(cd "${RECIPE_DIR}/../../verl-v1" && pwd)"
    elif [[ -d "${RECIPE_DIR}/../../verl/trainer/config" ]]; then
        VERL_ROOT="$(cd "${RECIPE_DIR}/../.." && pwd)"
    else
        echo "ERROR: cannot find a complete verl source tree in Ray working_dir." >&2
        echo "Submit from the repo root that contains both verl/ and recipe/." >&2
        exit 1
    fi
fi

# ================= cluster topology =================
# fully_async (like separate_async) splits GPUs into a TRAINER pool (also flips
# to rollout when idle = "hybrid engine") and a STANDALONE ROLLOUT pool (always
# generating). Both must be > 0.
#   trainer pool   -> trainer.{nnodes,n_gpus_per_node}
#   standalone pool-> actor_rollout_ref.rollout.{nnodes,n_gpus_per_node}
trainer_nnodes=${trainer_nnodes:-2}
n_gpus_training=${n_gpus_training:-8}
rollout_nnodes=${rollout_nnodes:-1}
n_gpus_rollout=${n_gpus_rollout:-8}

# ================= data / model =================
HF_MODEL_PATH=${HF_MODEL_PATH:-"/efs/data/models/Qwen3.5-9B"}
train_files=${train_files:-/efs/data/cua/rl/osworld_qwen35/train.parquet}
test_files=${test_files:-/efs/data/cua/rl/osworld_qwen35/test.parquet}
multi_turn_format=${multi_turn_format:-qwen3_coder}
use_chat_template_tools=${use_chat_template_tools:-True}

# ================= desktop env service =================
export DESKTOP_API_BASE_URL=${DESKTOP_API_BASE_URL:-http://172.31.13.38:2354}

# ================= rollout / agent loop =================
rollout_mode="async"
rollout_name=${rollout_name:-vllm}
rollout_logprobs_mode=${rollout_logprobs_mode:-raw_logprobs}
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
fi

tool_config_path=${tool_config_path:-${RECIPE_DIR}/tool_config.yaml}
agent_loop_config_path=${agent_loop_config_path:-${RECIPE_DIR}/agent.yaml}

# ================= algorithm =================
adv_estimator=grpo

max_turns=${max_turns:-50}
max_prompt_length=${max_prompt_length:-24567}
max_response_length=${max_response_length:-4096}
actor_lr=${actor_lr:-5e-6}
clip_ratio_low=${clip_ratio_low:-0.2}
clip_ratio_high=${clip_ratio_high:-0.28}
# turn_penalty_coef is NOT passed yet: it is not a field on verl AgentLoopConfig, so the
# override would crash at AgentLoopConfig instantiation. To enable: add `turn_penalty_coef: float = 0.0`
# to verl/workers/config/rollout.py AgentLoopConfig (+ rollout.yaml), then add this override to the
# agent block below:  +actor_rollout_ref.rollout.agent.turn_penalty_coef=${turn_penalty_coef}
turn_penalty_coef=${turn_penalty_coef:-0.1}
# Dr.GRPO advantage: reward - group_mean, WITHOUT dividing by std (norm_adv_by_std_in_grpo=False).
# grpo_adv_std_floor is inert when std-normalization is off.
norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo:-False}
grpo_adv_std_floor=${grpo_adv_std_floor:-0.1}
loss_agg_mode=${loss_agg_mode:-rollout-mean-token-sum-sqrt-norm}
loss_scale_factor=${loss_scale_factor:-55}

# V1 separate_async/fully_async assert data.train_batch_size == actor.ppo_mini_batch_size.
# This is the consumption batch (prompt groups per trainer step) AND the unit the
# streaming feeder dispatches into TransferQueue.
train_prompt_bsz=${train_prompt_bsz:-24}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-${train_prompt_bsz}}
n_resp_per_prompt=${n_resp_per_prompt:-8}
total_training_steps=${total_training_steps:-100000}
total_epochs=100000
test_freq=-1  # disabled: validation competes for desktop-env containers

# ---- V1 fully_async streaming / staleness controls ----
# Streaming: the feeder is the sole producer and fills the pipeline itself, so no
# warmup backlog (warmup only injects stale gs~0 prompts that age past budget).
num_warmup_batches=${num_warmup_batches:-0}
# Every N steps the trainer pushes new weights to the standalone rollout pool.
# Force 1 for on-policy debugging; otherwise rollout can lag the actor between
# sync points even when staleness_threshold=0.
parameter_sync_step=1
# Off-policy staleness budget (in parameter-sync units). Force 0 for on-policy
# debugging; raise this only when intentionally measuring async throughput.
staleness_threshold=0
# Seconds the feeder sleeps when the in-flight budget is full (avoids busy-wait).
feeder_poll_interval=${feeder_poll_interval:-1.0}
# Per-worker cap on concurrently-executing rollouts (event-loop / GIL pressure knob).
# Rollouts are dispatched one session at a time across the worker pool; total concurrency
# is num_workers * this. Keep it modest so a single AgentLoopWorker process isn't GIL-bound.
max_concurrent_rollouts_per_worker=${max_concurrent_rollouts_per_worker:-4}
# Store multimodal pixel tensors as bf16 in TransferQueue (~halves their RAM footprint in the
# storage-unit actors; the model consumes bf16 anyway). Set False to keep float32.
# Temporarily OFF: A/B test for the train<->infer logprob gap. bf16 pixel storage is a v1-only
# feature (pre-v1 didn't have it) and a suspect for the elevated uncertain-token divergence.
# NOTE: doubles image storage footprint -> only safe at the reduced bs=16.
multimodal_storage_bf16=${multimodal_storage_bf16:-True}
# TransferQueue total capacity in ENTRIES (rows + unique images), across all partitions/units.
# The default (100000) is too small here and causes ring-buffer EVICTION of still-referenced images
# -> "key ... not found in field 'image_grid_thw'" crashes. Size for the worst-case in-flight volume:
#   overshoot_budget(~96 prompts) * n(16) * max_turns(50) * 2 (train rows + rollout_images) ~= 154k.
# It's a count cap, not a preallocation, so headroom is cheap (actual RAM is bounded by the feeder
# staleness budget, not by this number).
tq_storage_size=${tq_storage_size:-500000}
# none: no trainer-side staleness gate — sample the oldest ready prompts and rely
# on the feeder budget + rollout correction (vs separate_async's default `drop`).
max_off_policy_strategy=${max_off_policy_strategy:-none}
# NOTE: must be a positive INT (replay_buffer asserts isinstance int AND > 0, so 0 is NOT allowed).
# Only used as a gate when max_off_policy_strategy != none; with `none` (our default) it is asserted
# but never read, so any positive int is fine. Kept a fixed int (decoupled from staleness_threshold)
# so a FLOAT staleness (e.g. 1.5) can't break this line via bash's integer-only $(()). If you switch
# to wait/drop, set it ~= ceil(staleness_threshold)+1.
max_off_policy_threshold=${max_off_policy_threshold:-1}

# Standalone rollout replicas must use a real weight-transfer checkpoint engine
# (separate_async/fully_async forbid the "naive" backend).
checkpoint_engine_backend=${checkpoint_engine_backend:-nccl}

# Rollout correction preset (same as the separate_async GUI recipe).
rollout_correction_bypass_mode=${rollout_correction_bypass_mode:-True}
rollout_correction_loss_type=${rollout_correction_loss_type:-ppo_clip}
rollout_correction_is=${rollout_correction_is:-null}
rollout_correction_rs=${rollout_correction_rs:-seq_mean_k3}
rollout_correction_rs_threshold=${rollout_correction_rs_threshold:-0.005}
case "${rollout_correction_bypass_mode}" in
    True|true|TRUE|1)
        actor_policy_loss_mode=${actor_policy_loss_mode:-bypass_mode}
        ;;
    *)
        actor_policy_loss_mode=${actor_policy_loss_mode:-vanilla}
        ;;
esac

# entropy_coeff=0 for this recipe, so keep the extra entropy/compiled-logits path
# disabled by default while stabilizing Qwen3.5 FSDP actor updates.
calculate_entropy=${calculate_entropy:-False}

# ---- Per-image dedup (opt-in; ppo/v1 untouched, enabled via subclass selection) ----
# On (image_dedup_enabled=True): selects a dedup-aware agent-loop manager + replay
#   buffer; each unique screenshot is stored once in rollout_images (keyed by SHA1),
#   rows carry only image_ids, refcount-GC'd as rows leave the replay buffer.
# `agent_loop_manager_class` is read dynamically (not a struct field) -> append with +.
image_dedup_enabled=${image_dedup_enabled:-True}
dedup_args=()
if [ "${image_dedup_enabled}" = "True" ]; then
    dedup_args=(
        +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.utils.transferqueue_image_dedup_v1.ImageDedupManagerTQ
        trainer.v1.sampler.custom_sampler.path="${VERL_ROOT}/verl/utils/transferqueue_image_dedup_v1.py"
        trainer.v1.sampler.custom_sampler.name=ImageDedupReplayBuffer
    )
fi

# ================= performance =================
# vLLM rollout tensor parallelism. TP=2 uses two GPUs per rollout engine.
infer_tp=${infer_tp:-1}
# Megatron training mesh. Defaults target the dense Qwen3.5-9B/27B path. For MoE
# checkpoints, override train_ep/train_etp explicitly.
train_tp=${train_tp:-${ACTOR_TP:-2}}
train_pp=${train_pp:-${ACTOR_PP:-1}}
train_cp=${train_cp:-${ACTOR_CP:-1}}
train_ep=${train_ep:-${ACTOR_EP:-1}}
train_etp=${train_etp:-${ACTOR_ETP:-1}}
megatron_all_offload=${megatron_all_offload:-True}
megatron_use_mbridge=${megatron_use_mbridge:-True}
megatron_vanilla_mbridge=${megatron_vanilla_mbridge:-True}
# vLLM's custom all-reduce can be faster, but TP>1 may hit CUDA/custom-allreduce
# compatibility issues on some clusters. Disable it by default for the debug recipe.
vllm_disable_custom_all_reduce=${vllm_disable_custom_all_reduce:-False}
actor_freeze_vision_tower=${actor_freeze_vision_tower:-True}
actor_use_torch_compile=${actor_use_torch_compile:-False}
actor_model_dtype=${actor_model_dtype:-bfloat16}
# Qwen3.5 Megatron currently uses BSHD/no-rmpad; packed THD is not the stable
# path for its native multimodal/GDN stack.
actor_use_dynamic_bsz=${actor_use_dynamic_bsz:-False}
model_use_remove_padding=${model_use_remove_padding:-False}
megatron_use_remove_padding=${megatron_use_remove_padding:-False}
model_use_fused_kernels=${model_use_fused_kernels:-False}
model_fused_kernel_backend=${model_fused_kernel_backend:-triton}
actor_ppo_max_token_len=${actor_ppo_max_token_len:-24000}
infer_ppo_max_token_len=${infer_ppo_max_token_len:-100000}

total_train_gpus=$((trainer_nnodes * n_gpus_training))
train_mesh_denom=$((train_tp * train_pp * train_cp))
if (( train_mesh_denom <= 0 || total_train_gpus % train_mesh_denom != 0 )); then
    echo "ERROR: invalid Megatron mesh: total_train_gpus=${total_train_gpus}, train_tp=${train_tp}, train_pp=${train_pp}, train_cp=${train_cp}" >&2
    exit 1
fi
if (( train_ep <= 0 || train_etp <= 0 )); then
    echo "ERROR: train_ep/train_etp must be positive, got train_ep=${train_ep}, train_etp=${train_etp}" >&2
    exit 1
fi
echo "[MEGATRON] train_gpus=${total_train_gpus} tp=${train_tp} pp=${train_pp} cp=${train_cp} ep=${train_ep} etp=${train_etp} offload=${megatron_all_offload}"

run_timestamp=$(TZ='Asia/Shanghai' date +%Y%m%d_%H%M%S)
project_name=${project_name:-v1_gui_agent_${run_timestamp}}
experiment_name=${experiment_name:-qwen35_9b_megatron_3nodes_8rollout_16train_v1_fully_async}
default_local_dir=${default_local_dir:-/efs/data/rl/checkpoints/${project_name}/${experiment_name}}
save_freq=${save_freq:-30}
resume_mode=${resume_mode:-auto}

# ---- One-shot FSDP/vLLM logprob root-cause probe ----
# Enabled by default for this debug script. It dumps the first actor micro-batch whose
# rollout-vs-FSDP RS-K3 crosses the mask threshold, including FSDP torch-reference
# selected logprobs, selected logits/top-k diagnostics, actor tags/global-step info,
# and a lightweight rank0 parameter fingerprint.
logprob_probe_enabled=${logprob_probe_enabled:-False}
if [ "${logprob_probe_enabled}" = "True" ]; then
    export VERL_LOGPROB_PROBE_DUMP=1
    export VERL_LOGPROB_PROBE_REF=1
    export VERL_LOGPROB_PROBE_MAX=${VERL_LOGPROB_PROBE_MAX:-1}
    export VERL_LOGPROB_PROBE_MIN_K3=${VERL_LOGPROB_PROBE_MIN_K3:-0.005}
    export VERL_LOGPROB_PROBE_TOPK=${VERL_LOGPROB_PROBE_TOPK:-5}
    export VERL_LOGPROB_PROBE_DIR=${VERL_LOGPROB_PROBE_DIR:-/efs/data/rl/logprob_probe_fsdp_${run_timestamp}}
    export VERL_LOGPROB_PROBE_LOGPROBS_MODE=${rollout_logprobs_mode}
    export VERL_LOGPROB_DEBUG_TOKENIZER=${HF_MODEL_PATH}
    echo "[LOGPROB_PROBE] enabled: dir=${VERL_LOGPROB_PROBE_DIR} min_k3=${VERL_LOGPROB_PROBE_MIN_K3}"
fi

# FSDP-only diagnostics. Megatron does not use this path; keep disabled unless
# explicitly re-testing the FSDP backend from this script.
fsdp_mem_debug=${fsdp_mem_debug:-False}
cuda_launch_blocking=${cuda_launch_blocking:-False}
ray_env_args=()
if [ "${fsdp_mem_debug}" = "True" ]; then
    export VERL_FSDP_MEM_DEBUG=1
    export VERL_FSDP_MEM_DEBUG_MAX_MICRO=${VERL_FSDP_MEM_DEBUG_MAX_MICRO:-4}
    export VERL_FSDP_MEM_DEBUG_RANKS=${VERL_FSDP_MEM_DEBUG_RANKS:-all}
    ray_env_args+=(
        +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FSDP_MEM_DEBUG=1
        +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FSDP_MEM_DEBUG_MAX_MICRO=${VERL_FSDP_MEM_DEBUG_MAX_MICRO}
        +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FSDP_MEM_DEBUG_RANKS=${VERL_FSDP_MEM_DEBUG_RANKS}
    )
fi
if [ "${cuda_launch_blocking}" = "True" ]; then
    export CUDA_LAUNCH_BLOCKING=1
    ray_env_args+=(+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_LAUNCH_BLOCKING=1)
fi

# ================= launch =================
# Hydra config uses `hydra.searchpath: file://verl/trainer/config` (relative to
# CWD), and the `recipe.*` agent-loop target must be importable; both rely on
# launching from the verl source root.
cd "${VERL_ROOT}"

python3 -m verl.trainer.main_ppo \
    "${ray_env_args[@]}" \
    model_engine=megatron \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=fully_async \
    trainer.v1.fully_async.num_warmup_batches=${num_warmup_batches} \
    trainer.v1.fully_async.parameter_sync_step=${parameter_sync_step} \
    trainer.v1.fully_async.staleness_threshold=${staleness_threshold} \
    trainer.v1.fully_async.feeder_poll_interval=${feeder_poll_interval} \
    trainer.v1.fully_async.max_concurrent_rollouts_per_worker=${max_concurrent_rollouts_per_worker} \
    trainer.v1.fully_async.multimodal_storage_bf16=${multimodal_storage_bf16} \
    trainer.v1.sampler.max_off_policy_threshold=${max_off_policy_threshold} \
    trainer.v1.sampler.max_off_policy_strategy=${max_off_policy_strategy} \
    transfer_queue.enable=True \
    transfer_queue.backend.storage_backend=SimpleStorage \
    transfer_queue.backend.SimpleStorage.num_data_storage_units=6 \
    transfer_queue.backend.SimpleStorage.total_storage_size=${tq_storage_size} \
    "${dedup_args[@]}" \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.grpo_adv_std_floor=${grpo_adv_std_floor} \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=${rollout_correction_bypass_mode} \
    algorithm.rollout_correction.loss_type=${rollout_correction_loss_type} \
    algorithm.rollout_correction.rollout_is=${rollout_correction_is} \
    algorithm.rollout_correction.rollout_rs=${rollout_correction_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_correction_rs_threshold} \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path="${HF_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${model_use_remove_padding} \
    actor_rollout_ref.model.use_fused_kernels=${model_use_fused_kernels} \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=${model_fused_kernel_backend} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.clip_grad=2.0 \
    'actor_rollout_ref.actor.checkpoint.load_contents=["model"]' \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=${actor_use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_torch_compile=${actor_use_torch_compile} \
    actor_rollout_ref.actor.megatron.use_mbridge=${megatron_use_mbridge} \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=${megatron_vanilla_mbridge} \
    actor_rollout_ref.actor.megatron.use_remove_padding=${megatron_use_remove_padding} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp} \
    actor_rollout_ref.actor.megatron.param_offload=${megatron_all_offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${megatron_all_offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${megatron_all_offload} \
    actor_rollout_ref.actor.megatron.dtype=${actor_model_dtype} \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.freeze_vision_tower=${actor_freeze_vision_tower} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.loss_scale_factor=${loss_scale_factor} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=${calculate_entropy} \
    +actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=${actor_policy_loss_mode} \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=${rollout_correction_bypass_mode} \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=${rollout_correction_loss_type} \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=${rollout_correction_is} \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs=${rollout_correction_rs} \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs_threshold=${rollout_correction_rs_threshold} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${actor_use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_mbridge=${megatron_use_mbridge} \
    actor_rollout_ref.ref.megatron.vanilla_mbridge=${megatron_vanilla_mbridge} \
    actor_rollout_ref.ref.megatron.use_remove_padding=${megatron_use_remove_padding} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp} \
    actor_rollout_ref.ref.megatron.param_offload=${megatron_all_offload} \
    actor_rollout_ref.ref.megatron.dtype=${actor_model_dtype} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.nnodes=${rollout_nnodes} \
    actor_rollout_ref.rollout.n_gpus_per_node=${n_gpus_rollout} \
    actor_rollout_ref.rollout.checkpoint_engine.backend=${checkpoint_engine_backend} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.logprobs_mode=${rollout_logprobs_mode} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${actor_use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=${vllm_disable_custom_all_reduce} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=${multi_turn_format} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${tool_config_path} \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${agent_loop_config_path} \
    actor_rollout_ref.rollout.agent.default_agent_loop=gui_agent \
    actor_rollout_ref.rollout.agent.num_workers=32 \
    actor_rollout_ref.rollout.agent.use_chat_template_tools=${use_chat_template_tools} \
    trainer.logger='["console", "mlflow"]' \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=null \
    trainer.balance_batch=False \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.total_training_steps="${total_training_steps}" \
    trainer.val_before_train=False \
    trainer.test_freq="${test_freq}" \
    trainer.save_freq="${save_freq}" \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.resume_mode="${resume_mode}" \
    trainer.nnodes="${trainer_nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    "$@"
