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
# Fully-Async GUI Agent (Computer-Use Agent) PPO training.
# 8B model, DISAGGREGATED three-node topology:
#   - 1 node  dedicated to rollout  (8 GPUs: vLLM + agent loops)
#   - 2 nodes dedicated to training (2 x 8 = 16 GPUs: FSDP2 actor/ref, HSDP:
#     shard within each node over NVLink, replicate across the 2 nodes)
# This separates rollout-side memory (image banks / object store / agent-loop
# CPU preprocessing) from training-side memory (weights/optimizer/activations),
# and gives rollout a whole node's GPUs+CPUs. See the 8_8 script for the 1+1
# variant and the 6_2 script for the colocated variant.
#
# IMPORTANT (placement): only GPU-bound actors are auto-pinned by their
# placement groups. The CPU-bound FullyAsyncRollouter, the 40 AgentLoopWorkers
# and the GlobalRequestLoadBalancer are NOT GPU-pinned and Ray would spread them
# across all nodes by default. Placement is constrained at the cluster level via
# custom Ray resources: tag the rollout node `rollout_node` and the 2 train
# nodes `train_node` at `ray start`, and export VERL_ROLLOUT_NODE_RESOURCE /
# VERL_TRAIN_NODE_RESOURCE so verl pins each role to the right node(s).
#
# Prerequisites:
#   1. A running desktop environment service accessible via HTTP (endpoints:
#      /session/create, /session/{id}/step, /session/{id}/evaluate,
#      /session/{id}/close). Set DESKTOP_API_BASE_URL to its endpoint.
#   2. A VLM checkpoint (e.g. Qwen2.5-VL-7B-Instruct).
#   3. A parquet dataset with columns:
#        - prompt (list of chat messages)
#        - extra_info.task_id (desktop task identifier)
#        - extra_info.question (user query for the task)

set -xeuo pipefail
export HYDRA_FULL_ERROR=1
export HF_HOME="/efs/data/hf"

# ================= process resource limits =================
# Raise nofile so busy actors (many aiohttp requests + gRPC connections)
# cannot hit EMFILE, which has been observed to trigger SIGABRT inside
# libuv's ``uv__epoll_ctl_flush`` on the Ray core-worker IO thread.
# Enable unlimited core dumps so the next crash leaves a file we can
# inspect with gdb.
ulimit -n 1048576 || true
ulimit -c unlimited || true

# Make sure the env vars we set below are propagated to all Ray workers.
# NOTE: runtime_env.yaml already includes gRPC/Ray tweaks that address
# the same SIGABRT; this script-side ulimit covers the case where a
# worker bypasses runtime_env (e.g. driver-local processes).
# export VERL_LOGGING_LEVEL=DEBUG
# WandB / Weave config. Set WANDB_API_KEY externally; optionally WANDB_BASE_URL
# for on-prem wandb. WEAVE_PROJECT defaults to the verl project_name.
export WANDB_API_KEY=${WANDB_API_KEY:-}
# Ray defaults to uvloop when installed; disable it before Ray workers start to
# avoid uvloop/aiohttp fd ownership bugs under timeout/retry-heavy rollouts.
export RAY_USE_UVLOOP=${RAY_USE_UVLOOP:-0}
# Reduce memory fragmentation (helps with the 27GB reserved-but-unallocated).
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
# ================= paths =================
# RECIPE_DIR is the directory containing this script. VERL_ROOT must point at
# the verl source tree so Hydra's ``hydra.searchpath: file://verl/trainer/config``
# can resolve. In Ray jobs, infer it from the packaged working_dir instead of a
# node-local /root/verl checkout.
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${VERL_ROOT:-}" ]]; then
    if [[ -d "verl/trainer/config" ]]; then
        VERL_ROOT="$(pwd)"
    elif [[ -d "${RECIPE_DIR}/../../verl/trainer/config" ]]; then
        VERL_ROOT="$(cd "${RECIPE_DIR}/../.." && pwd)"
    else
        echo "ERROR: cannot find a complete verl source tree in Ray working_dir." >&2
        echo "Submit from the repo root that contains both verl/ and recipe/." >&2
        exit 1
    fi
fi

# ================= cluster topology =================
NNODES=${NNODES:-3}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Disaggregated resource split: 1 node rollout, 2 nodes training.
# rollout_nnodes=1 x n_gpus_rollout=8   -> all 8 GPUs of one node serve vLLM.
# trainer_nnodes=2 x n_gpus_training=8  -> 16 training GPUs across 2 nodes.
# The trainer pool spec [8, 8] is two 8-GPU STRICT_PACK placement groups -> one
# per train node; the rollout pool is one 8-GPU group on the rollout node.
n_gpus_rollout=${n_gpus_rollout:-8}
n_gpus_training=${n_gpus_training:-8}
rollout_nnodes=${rollout_nnodes:-1}
trainer_nnodes=${trainer_nnodes:-2}

# ================= data / model =================
# HF_MODEL_PATH=${HF_MODEL_PATH:-"/efs/data/cua/runs/0525e-8b-osworld-plus-new/v0-20260525-160300/checkpoint-810-merged"}
# HF_MODEL_PATH=${HF_MODEL_PATH:-"/efs/data/cua/runs/0608f-general-osworld-plus-new-agentnet/v0-20260608-205226/checkpoint-1500-merged"}
HF_MODEL_PATH=${HF_MODEL_PATH:-"/efs/train/hf_models/qwen3-vl-8b-instruct"}
train_files=${train_files:-/efs/data/cua/rl/osworld/train.parquet}
test_files=${test_files:-/efs/data/cua/rl/osworld/test.parquet}

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
max_prompt_length=${max_prompt_length:-20480}
max_response_length=${max_response_length:-4096}
actor_lr=${actor_lr:-5e-6}
clip_ratio_low=${clip_ratio_low:-0.2}
clip_ratio_high=${clip_ratio_high:-0.28}
turn_penalty_coef=${turn_penalty_coef:-0.1}
norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo:-True}
grpo_adv_std_floor=${grpo_adv_std_floor:-0.1}
fail_loop_no_change_adv_coef=${fail_loop_no_change_adv_coef:-0.02}
loss_agg_mode=${loss_agg_mode:-rollout-mean-token-sum-sqrt-norm}
loss_scale_factor=${loss_scale_factor:-null}

# Fully-async uses gen_batch_size=1 (streaming single-sample generation).
train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=${n_resp_per_prompt:-16}
train_prompt_mini_bsz=8
require_batches=${require_batches:-1}
total_rollout_steps=${total_rollout_steps:-100000}
total_epochs=100000
test_freq=-1


# Async stream pipeline with partial rollout (see fully_async README).
staleness_threshold=${staleness_threshold:-1}
trigger_parameter_sync_step=${trigger_parameter_sync_step:-2}
partial_rollout=${partial_rollout:-True}

# Rollout correction preset: RolloutCorrectionConfig.bypass_ppo_clip_geo_rs().
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

# Entropy is computed for logging only; keep entropy_coeff=0 to avoid changing the objective.
calculate_entropy=${calculate_entropy:-True}

# Hard cap on in-flight rollout trajectories / desktop-env sessions.
# Sample-level in-flight capacity is derived from max_required_samples; do not
# add an extra sample cap here, otherwise long-tail samples can block later
# samples from filling newly available env slots.
# One full rollout node (8 GPUs) backs these trajectories.
max_concurrent_rollouts=${max_concurrent_rollouts:-144}
# Validation can launch the whole test set (~300 tasks) at once; keep its env
# session pressure separate from training throughput.
max_concurrent_eval_rollouts=${max_concurrent_eval_rollouts:-150}
loop_no_change_repeat_min=${loop_no_change_repeat_min:-5}
loop_no_change_diff_threshold=${loop_no_change_diff_threshold:-2.0}

# ================= performance =================
infer_tp=${infer_tp:-1}
actor_param_offload=${actor_param_offload:-False}
actor_optimizer_offload=${actor_optimizer_offload:-False}
actor_freeze_vision_tower=${actor_freeze_vision_tower:-True}
ref_offload=${ref_offload:-False}
# FSDP shard-group size = GPUs per node. With 2 train nodes x 8 GPUs this gives
# HSDP: shard the 8B params/grads within each node over NVLink, and replicate
# (data-parallel + grad all-reduce) across the 2 nodes. Keeps the per-layer
# all-gather node-local instead of sharding across the slow inter-node link.
fsdp_size=${n_gpus_training}

# Max packed-sequence length per GPU per micro-batch (dynamic_bsz on).
# With Qwen3-VL-8B + FSDP2, a 64k packed sequence OOMs on 140GB even with
# param/optimizer offload, because the (seq_len^2) attention activations plus
# FSDP all-gather of the 8B params/grads exceed what fits. Keeping this at
# ~(max_prompt+max_response) is safer; scale up only if backward fits.
actor_ppo_max_token_len=50000
infer_ppo_max_token_len=100000

# Timestamp in UTC+8 (Asia/Shanghai), independent of the host timezone.
run_timestamp=$(TZ='Asia/Shanghai' date +%Y%m%d_%H%M%S)
project_name=${project_name:-fully_async_gui_agent_${run_timestamp}}
# project_name="fully_async_gui_agent_20260621_044118"
experiment_name=${experiment_name:-qwen3vl_8b_3nodes_8rollout_16train_async}
default_local_dir=${default_local_dir:-/efs/data/rl/checkpoints/${project_name}/${experiment_name}}
save_freq=30

# ================= launch =================
# Hydra's config uses ``hydra.searchpath: file://verl/trainer/config`` which is
# resolved relative to CWD, so chdir to the verl source root before launching.
cd "${VERL_ROOT}"

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.grpo_adv_std_floor=${grpo_adv_std_floor} \
    algorithm.fail_loop_no_change_adv_coef=${fail_loop_no_change_adv_coef} \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${HF_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    'actor_rollout_ref.actor.checkpoint.load_contents=["model"]' \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload} \
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
    actor_rollout_ref.actor.grad_clip=2.0 \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=${actor_policy_loss_mode} \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=${rollout_correction_bypass_mode} \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=${rollout_correction_loss_type} \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=${rollout_correction_is} \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs=${rollout_correction_rs} \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs_threshold=${rollout_correction_rs_threshold} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.logprobs_mode=${rollout_logprobs_mode} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${tool_config_path} \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${agent_loop_config_path} \
    actor_rollout_ref.rollout.agent.num_workers=32 \
    actor_rollout_ref.rollout.agent.turn_penalty_coef=${turn_penalty_coef} \
    actor_rollout_ref.rollout.agent.loop_no_change_repeat_min=${loop_no_change_repeat_min} \
    actor_rollout_ref.rollout.agent.loop_no_change_diff_threshold=${loop_no_change_diff_threshold} \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=${rollout_correction_bypass_mode} \
    algorithm.rollout_correction.loss_type=${rollout_correction_loss_type} \
    algorithm.rollout_correction.rollout_is=${rollout_correction_is} \
    algorithm.rollout_correction.rollout_rs=${rollout_correction_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_correction_rs_threshold} \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    +async_training.max_concurrent_rollouts="${max_concurrent_rollouts}" \
    +async_training.max_concurrent_eval_rollouts="${max_concurrent_eval_rollouts}" \
    ++async_training.image_refs.enabled=True \
    trainer.logger='["console", "mlflow"]' \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    trainer.balance_batch=False \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.val_before_train=False \
    trainer.test_freq="${test_freq}" \
    trainer.save_freq="${save_freq}" \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.resume_mode=auto \
    trainer.nnodes="${trainer_nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${rollout_nnodes}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    "$@"
