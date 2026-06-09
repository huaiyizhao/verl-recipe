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
# Fully-Async GUI Agent OPD training.
#
# This is a self-contained OPD variant of run_fully_async_gui_agent_4_4.sh.
# Defaults reserve 1 GPU for rollout and 1 GPU for the teacher model. Set
# TEACHER_MODEL to a frozen teacher checkpoint before launching.

set -xeuo pipefail
export HYDRA_FULL_ERROR=1
export HF_HOME=${HF_HOME:-"/efs/data/hf"}

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
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Fully-async resource split: rollout, teacher, and training pools.
n_gpus_rollout=${n_gpus_rollout:-1}
rollout_nnodes=${rollout_nnodes:-1}
teacher_nnodes=${teacher_nnodes:-${NNODES}}
teacher_n_gpus_per_node=${teacher_n_gpus_per_node:-${TEACHER_WORLD_SIZE:-1}}
trainer_nnodes=${trainer_nnodes:-1}
if [ -z "${n_gpus_training:-}" ]; then
    n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout - teacher_n_gpus_per_node))
    if [ "${n_gpus_training}" -lt 1 ]; then
        n_gpus_training=1
    fi
fi

# ================= data / model =================
HF_MODEL_PATH=${HF_MODEL_PATH:-"/efs/data/cua/runs/0525e-8b-osworld-plus-new/v0-20260525-160300/checkpoint-810-merged"}
train_files=${train_files:-/efs/data/cua/rl/osworld/train.parquet}
test_files=${test_files:-/efs/data/cua/rl/osworld/test.parquet}

if [ -z "${TEACHER_MODEL:-}" ]; then
    echo "TEACHER_MODEL must point to the frozen teacher checkpoint for OPD." >&2
    exit 1
fi

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

max_turns=${max_turns:-30}
max_prompt_length=${max_prompt_length:-24576}
max_response_length=${max_response_length:-8192}
actor_lr=${actor_lr:-1e-6}
clip_ratio_low=${clip_ratio_low:-0.2}
clip_ratio_high=${clip_ratio_high:-0.28}
turn_penalty_coef=${turn_penalty_coef:-0.1}
loss_agg_mode=${loss_agg_mode:-seq-mean-token-sum-norm}
loss_scale_factor=${loss_scale_factor:-${max_response_length}}

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=${n_resp_per_prompt:-8}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-8}
require_batches=${require_batches:-1}
total_rollout_steps=${total_rollout_steps:-100000}
total_epochs=${total_epochs:-100000}
test_freq=${test_freq:--20}

# Async stream pipeline with partial rollout.
staleness_threshold=${staleness_threshold:-1}
trigger_parameter_sync_step=${trigger_parameter_sync_step:-4}
partial_rollout=${partial_rollout:-True}

# Rollout correction preset: RolloutCorrectionConfig.bypass_ppo_clip_geo_rs().
rollout_correction_bypass_mode=${rollout_correction_bypass_mode:-True}
rollout_correction_loss_type=${rollout_correction_loss_type:-ppo_clip}
rollout_correction_is=${rollout_correction_is:-null}
rollout_correction_rs=${rollout_correction_rs:-seq_mean_k1}
rollout_correction_rs_threshold=${rollout_correction_rs_threshold:-0.999_1.001}

calculate_entropy=${calculate_entropy:-True}
max_concurrent_rollouts=${max_concurrent_rollouts:-16}

# ================= OPD distillation =================
teacher_tp=${teacher_tp:-${teacher_n_gpus_per_node}}
teacher_gpu_mem_util=${teacher_gpu_mem_util:-0.8}
teacher_max_model_len=${teacher_max_model_len:-32768}

# Use k1/k3 estimator modes for sampled-token OPD. forward_kl_topk is supported
# by the same code path but is top-k forward-KL distillation, not OPD estimator.
distillation_loss_mode=${distillation_loss_mode:-k3}
distillation_use_policy_gradient=${distillation_use_policy_gradient:-True}
distillation_use_task_rewards=${distillation_use_task_rewards:-True}
distillation_loss_coef=${distillation_loss_coef:-0.1}
distillation_loss_max_clamp=${distillation_loss_max_clamp:-5.0}
distillation_log_prob_min_clamp=${distillation_log_prob_min_clamp:-null}
distillation_topk=${distillation_topk:-32}

# ================= performance =================
infer_tp=${infer_tp:-1}
actor_param_offload=${actor_param_offload:-False}
actor_optimizer_offload=${actor_optimizer_offload:-True}
actor_freeze_vision_tower=${actor_freeze_vision_tower:-True}
ref_offload=${ref_offload:-True}
fsdp_size=${fsdp_size:-${n_gpus_training}}

actor_ppo_max_token_len=${actor_ppo_max_token_len:-48000}
infer_ppo_max_token_len=${infer_ppo_max_token_len:-96000}

project_name=${project_name:-fully_async_gui_agent_opd_0526}
experiment_name=${experiment_name:-qwen3vl_8b_fsdp_async_opd}

# ================= launch =================
cd "${VERL_ROOT}"

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=False \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload} \
    actor_rollout_ref.actor.freeze_vision_tower=${actor_freeze_vision_tower} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.loss_scale_factor=${loss_scale_factor} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=${calculate_entropy} \
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
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${tool_config_path} \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${agent_loop_config_path} \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.turn_penalty_coef=${turn_penalty_coef} \
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
    ++async_training.image_refs.enabled=True \
    distillation.enabled=True \
    distillation.n_gpus_per_node="${teacher_n_gpus_per_node}" \
    distillation.nnodes="${teacher_nnodes}" \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${teacher_tp}" \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${teacher_gpu_mem_util}" \
    distillation.teacher_models.teacher_model.inference.max_model_len="${teacher_max_model_len}" \
    distillation.distillation_loss.loss_mode="${distillation_loss_mode}" \
    distillation.distillation_loss.topk="${distillation_topk}" \
    distillation.distillation_loss.use_policy_gradient="${distillation_use_policy_gradient}" \
    distillation.distillation_loss.use_task_rewards="${distillation_use_task_rewards}" \
    distillation.distillation_loss.distillation_loss_coef="${distillation_loss_coef}" \
    distillation.distillation_loss.loss_max_clamp="${distillation_loss_max_clamp}" \
    distillation.distillation_loss.log_prob_min_clamp="${distillation_log_prob_min_clamp}" \
    trainer.logger='["console", "mlflow"]' \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    trainer.balance_batch=False \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.val_before_train=False \
    trainer.test_freq="${test_freq}" \
    trainer.save_freq="${test_freq}" \
    trainer.resume_mode=disable \
    trainer.nnodes="${trainer_nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${rollout_nnodes}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    "$@"
