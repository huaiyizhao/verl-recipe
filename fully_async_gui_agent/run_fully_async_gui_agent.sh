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
# ================= paths =================
# RECIPE_DIR is the directory containing this script (portable, no matter where
# the script is invoked from). VERL_ROOT must point at the verl source tree so
# that Hydra's ``hydra.searchpath: file://verl/trainer/config`` (relative to
# CWD) can resolve. Override VERL_ROOT if you use a different verl checkout.
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT=${VERL_ROOT:-/root/verl}

# ================= cluster topology =================
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Fully-async resource split: rollout vs training GPUs.
n_gpus_rollout=${n_gpus_rollout:-4}
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

export HYDRA_FULL_ERROR=1
# export VERL_LOGGING_LEVEL=DEBUG
# WandB / Weave config. Set WANDB_API_KEY externally; optionally WANDB_BASE_URL
# for on-prem wandb. WEAVE_PROJECT defaults to the verl project_name.
export WANDB_API_KEY=${WANDB_API_KEY:-}
# Reduce memory fragmentation (helps with the 27GB reserved-but-unallocated).
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

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

max_turns=${max_turns:-20}
max_prompt_length=${max_prompt_length:-24000}
max_response_length=${max_response_length:-8192}
actor_lr=${actor_lr:-1e-6}

# Fully-async uses gen_batch_size=1 (streaming single-sample generation).
train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=${n_resp_per_prompt:-8}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-2}
require_batches=${require_batches:-1}
total_rollout_steps=${total_rollout_steps:-1000}
total_epochs=200
test_freq=5


# Async stream pipeline with partial rollout (see fully_async README).
staleness_threshold=${staleness_threshold:-0}
trigger_parameter_sync_step=${trigger_parameter_sync_step:-1}
partial_rollout=${partial_rollout:-False}

# Hard cap on in-flight rollouts. The desktop-env service only allows a
# limited number of concurrent sessions (e.g. 32), so we must throttle the
# rollouter here to avoid flooding the backend.
max_concurrent_rollouts=${max_concurrent_rollouts:-16}

# ================= performance =================
infer_tp=${infer_tp:-1}
actor_offload=${actor_offload:-True}
ref_offload=${ref_offload:-True}
fsdp_size=4

# Max packed-sequence length per GPU per micro-batch (dynamic_bsz on).
# With Qwen3-VL-8B + FSDP2, a 64k packed sequence OOMs on 140GB even with
# param/optimizer offload, because the (seq_len^2) attention activations plus
# FSDP all-gather of the 8B params/grads exceed what fits. Keeping this at
# ~(max_prompt+max_response) is safer; scale up only if backward fits.
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3 / 2))

project_name=${project_name:-fully_async_gui_agent}
experiment_name=${experiment_name:-qwen3vl_8b_fsdp_async}

# ================= launch =================
# Hydra's config uses ``hydra.searchpath: file://verl/trainer/config`` which is
# resolved relative to CWD, so chdir to the verl source root before launching.
cd "${VERL_ROOT}"

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=${adv_estimator} \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
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
    actor_rollout_ref.rollout.agent.num_workers=4 \
    algorithm.use_kl_in_reward=False \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    +async_training.max_concurrent_rollouts="${max_concurrent_rollouts}" \
    trainer.logger='["console", "wandb"]' \
    actor_rollout_ref.rollout.trace.backend=weave \
    actor_rollout_ref.rollout.trace.token2text=True \
    trainer.balance_batch=False \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.val_before_train=False \
    trainer.test_freq="${test_freq}" \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    "$@"
