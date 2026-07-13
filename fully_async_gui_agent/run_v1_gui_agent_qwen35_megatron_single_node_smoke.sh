#!/usr/bin/env bash
# Single-node smoke test for Qwen3.5 GUI-agent Megatron training.
#
# Run this directly on one machine with the patched /root/verl checkout. It
# forces Ray to use a local cluster and splits 8 local GPUs into:
#   - 4 GPUs for Megatron actor/ref training
#   - 4 GPUs for standalone vLLM rollout

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RAY_ADDRESS=${RAY_ADDRESS:-local}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

export HF_MODEL_PATH=${HF_MODEL_PATH:-/efs/data/models/Qwen3.5-9B}
export train_files=${train_files:-/efs/data/cua/rl/osworld_qwen35/train.parquet}
export test_files=${test_files:-/efs/data/cua/rl/osworld_qwen35/test.parquet}

trainer_nnodes=${trainer_nnodes:-1} \
n_gpus_training=${n_gpus_training:-4} \
rollout_nnodes=${rollout_nnodes:-1} \
n_gpus_rollout=${n_gpus_rollout:-4} \
train_tp=${train_tp:-2} \
train_pp=${train_pp:-1} \
train_cp=${train_cp:-1} \
train_ep=${train_ep:-1} \
train_etp=${train_etp:-1} \
infer_tp=${infer_tp:-1} \
train_prompt_bsz=${train_prompt_bsz:-2} \
train_prompt_mini_bsz=${train_prompt_mini_bsz:-2} \
n_resp_per_prompt=${n_resp_per_prompt:-2} \
max_turns=${max_turns:-1} \
max_response_length=${max_response_length:-512} \
total_training_steps=${total_training_steps:-1} \
save_freq=${save_freq:-1000000} \
project_name=${project_name:-v1_gui_agent_single_node_smoke} \
experiment_name=${experiment_name:-qwen35_9b_megatron_single_node_smoke} \
bash "${SCRIPT_DIR}/run_v1_gui_agent_qwen35_fully_async.sh" \
    +ray_kwargs.ray_init.address=local \
    "$@"
