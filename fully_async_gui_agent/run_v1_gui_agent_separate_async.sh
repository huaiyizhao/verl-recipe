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
# GUI Agent (Computer-Use Agent) PPO training on the **V1 trainer**
# (separate_async mode, TransferQueue-native).
#
# This is the V1 port of run_fully_async_gui_agent_8_8.sh. The agent loop,
# tools, data, model and desktop-env settings are unchanged; only the async
# data-link is different: instead of the experimental fully_async_policy
# (Rollouter + Ray MessageQueue + Trainer actors), this uses the upstream V1
# `separate_async` trainer with a TransferQueue data plane + replay buffer.
#
# Config mapping from the old fully_async_policy knobs:
#   entry            verl.experimental.fully_async_policy.fully_async_main
#                  → verl.trainer.main_ppo  (trainer.use_v1=True)
#   async_training.trigger_parameter_sync_step
#                  → trainer.v1.separate_async.parameter_sync_step
#   async_training.staleness_threshold
#                  → trainer.v1.sampler.max_off_policy_threshold
#   async_training.partial_rollout / require_batches / max_concurrent_rollouts
#                  → (no direct V1 knob; replay buffer + warmup batches instead)
#   rollout.{nnodes,n_gpus_per_node}   (top-level standalone rollout pool)
#                  → actor_rollout_ref.rollout.{nnodes,n_gpus_per_node}
#   rollout.total_rollout_steps
#                  → trainer.total_training_steps
#
# Prerequisites (unchanged):
#   1. A running desktop-env service (DESKTOP_API_BASE_URL).
#   2. A VLM checkpoint (e.g. Qwen3-VL-8B-Instruct).
#   3. A parquet dataset with prompt / extra_info.task_id / extra_info.question.
#   4. A verl install at upstream/main level (the V1 trainer must exist) plus
#      `transfer_queue` (TransferQueue==0.1.8) on EVERY Ray node.

set -xeuo pipefail
export HYDRA_FULL_ERROR=1
export HF_HOME="/efs/data/hf"

# ================= process resource limits =================
ulimit -n 1048576 || true
ulimit -c unlimited || true

export WANDB_API_KEY=${WANDB_API_KEY:-}
export RAY_USE_UVLOOP=${RAY_USE_UVLOOP:-0}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

# ================= paths =================
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT=${VERL_ROOT:-/root/verl}

# ================= cluster topology =================
# V1 separate_async splits GPUs into a TRAINER pool (which also flips to rollout
# when idle = "hybrid engine") and a STANDALONE ROLLOUT pool (always generating).
#   trainer pool   -> trainer.{nnodes,n_gpus_per_node}
#   standalone pool-> actor_rollout_ref.rollout.{nnodes,n_gpus_per_node}
# Both must be > 0 in separate_async. Tune the split for your cluster.
trainer_nnodes=${trainer_nnodes:-2}
n_gpus_training=${n_gpus_training:-4}
rollout_nnodes=${rollout_nnodes:-2}
n_gpus_rollout=${n_gpus_rollout:-4}

# ================= data / model =================
HF_MODEL_PATH=${HF_MODEL_PATH:-"Qwen/Qwen3-VL-8B-Instruct"}
train_files=${train_files:-/efs/data/cua/rl/osworld/train.parquet}
test_files=${test_files:-/efs/data/cua/rl/osworld/test.parquet}

# ================= desktop env service =================
export DESKTOP_API_BASE_URL=${DESKTOP_API_BASE_URL:-http://10.192.64.238:2354}

# ================= rollout / agent loop =================
rollout_mode="async"
rollout_name=${rollout_name:-vllm}
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
fi

tool_config_path=${tool_config_path:-${RECIPE_DIR}/tool_config.yaml}
agent_loop_config_path=${agent_loop_config_path:-${RECIPE_DIR}/agent.yaml}

# ================= algorithm =================
adv_estimator=grpo

max_turns=${max_turns:-50}
max_prompt_length=${max_prompt_length:-20000}
max_response_length=${max_response_length:-8192}
actor_lr=${actor_lr:-1e-6}
turn_penalty_coef=${turn_penalty_coef:-0.1}
loss_agg_mode=${loss_agg_mode:-seq-mean-token-sum-norm}
loss_scale_factor=${loss_scale_factor:-${max_response_length}}

# V1 separate_async asserts data.train_batch_size == actor.ppo_mini_batch_size
# (one mini-batch consumed per step, streamed from the replay buffer).
train_prompt_bsz=${train_prompt_bsz:-6}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-${train_prompt_bsz}}
n_resp_per_prompt=${n_resp_per_prompt:-8}
total_training_steps=${total_training_steps:-1000}
total_epochs=200
test_freq=-1  # disabled: validation competes for desktop-env containers

# ---- V1 async / staleness controls ----
# Warmup batches primed before the training loop (keeps the replay buffer fed).
num_warmup_batches=${num_warmup_batches:-4}
# Every N steps the trainer pushes new weights to the standalone rollout pool.
parameter_sync_step=${parameter_sync_step:-2}
# Max model-versions a trajectory may span before it is dropped/waited.
max_off_policy_threshold=${max_off_policy_threshold:-1}
max_off_policy_strategy=${max_off_policy_strategy:-drop}

# Standalone rollout replicas must use a real weight-transfer checkpoint engine
# (separate_async forbids the "naive" backend).
checkpoint_engine_backend=${checkpoint_engine_backend:-nccl}

# ---- Per-image dedup (opt-in; ppo/v1 untouched, enabled via subclass selection) ----
# Off (default): screenshots stored inline per row (correct, memory-heavy).
# On (image_dedup_enabled=True): selects a dedup-aware agent-loop manager + replay
#   buffer (subclasses outside ppo/v1). Each unique screenshot is stored once in
#   rollout_images (keyed by SHA1); rows carry only image_ids, resolved inside the
#   worker on consume and refcount-GC'd as rows leave the replay buffer.
image_dedup_enabled=${image_dedup_enabled:-False}
dedup_args=()
if [ "${image_dedup_enabled}" = "True" ]; then
    dedup_args=(
        actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.utils.transferqueue_image_dedup_v1.ImageDedupManagerTQ
        trainer.v1.sampler.custom_sampler.path="${VERL_ROOT}/verl/utils/transferqueue_image_dedup_v1.py"
        trainer.v1.sampler.custom_sampler.name=ImageDedupReplayBuffer
    )
fi

# ================= performance =================
infer_tp=${infer_tp:-1}
actor_offload=${actor_offload:-False}
ref_offload=${ref_offload:-False}
fsdp_size=${n_gpus_training}
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3 / 2))

project_name=${project_name:-v1_gui_agent}
experiment_name=${experiment_name:-qwen3vl_8b_separate_async}

# ================= launch =================
# Hydra config uses `hydra.searchpath: file://verl/trainer/config` (relative to
# CWD), and the `recipe.*` agent-loop target must be importable; both rely on
# launching from the verl source root (same as the fully_async scripts).
cd "${VERL_ROOT}"

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=${num_warmup_batches} \
    trainer.v1.separate_async.parameter_sync_step=${parameter_sync_step} \
    trainer.v1.sampler.max_off_policy_threshold=${max_off_policy_threshold} \
    trainer.v1.sampler.max_off_policy_strategy=${max_off_policy_strategy} \
    transfer_queue.enable=True \
    transfer_queue.backend.storage_backend=SimpleStorage \
    "${dedup_args[@]}" \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${HF_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.loss_scale_factor=${loss_scale_factor} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.nnodes=${rollout_nnodes} \
    actor_rollout_ref.rollout.n_gpus_per_node=${n_gpus_rollout} \
    actor_rollout_ref.rollout.checkpoint_engine.backend=${checkpoint_engine_backend} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${tool_config_path} \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${agent_loop_config_path} \
    actor_rollout_ref.rollout.agent.default_agent_loop=gui_agent \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.turn_penalty_coef=${turn_penalty_coef} \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    trainer.logger='["console", "mlflow"]' \
    trainer.balance_batch=False \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.total_training_steps="${total_training_steps}" \
    trainer.val_before_train=False \
    trainer.test_freq="${test_freq}" \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.nnodes="${trainer_nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    "$@"
