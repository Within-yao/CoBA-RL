#!/usr/bin/env bash
set -x
set -euxo pipefail
ulimit -n 65535

# Auto-detect GPU count and configure
detect_gpus() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
        GPU_LIST="$CUDA_VISIBLE_DEVICES"
    else
        if command -v nvidia-smi &> /dev/null; then
            GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
            GPU_LIST=$(seq -s, 0 $((GPU_COUNT - 1)))
        else
            echo "Warning: nvidia-smi not found and CUDA_VISIBLE_DEVICES not set. Defaulting to 8 GPUs."
            GPU_COUNT=8
            GPU_LIST="0,1,2,3,4,5,6,7"
        fi
    fi
    echo "Detected $GPU_COUNT GPU(s): $GPU_LIST"
    export AUTO_DETECTED_GPU_COUNT=$GPU_COUNT
    export AUTO_DETECTED_GPU_LIST=$GPU_LIST
}

detect_gpus

export VERL_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1

export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DISABLE_GPU_MONITOR=1
export RAY_DEBUG_POST_MORTEM=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"

export RAY_worker_register_timeout_seconds=600
export RAY_TASK_MAX_RETRIES=3
export RAY_memory=100000000000
export RAY_object_store_memory=50000000000

#################### Multi-node Parameters ####################
# Number of nodes (set to 1 for single node training)
# For 32 GPUs across 4 nodes: N_NODE=4, n_gpus_per_node=8
N_NODE=${N_NODE:-1}
n_gpus_per_node=${n_gpus_per_node:-8}
#################################################################

# Base path configuration
CUR_MODEL="Qwen2.5-7B-Instruct"
total_epochs=15

PROJECT_NAME="coba-rl"

# Get current directory as working directory
WORKING_DIR="$(pwd)"
CONFIG_PATH="${WORKING_DIR}/recipe/coba_rl/config"
echo "WORKING_DIR: ${WORKING_DIR}"

# Base paths (modify these according to your environment)
BASE_OUTPUT_DIR="${WORKING_DIR}/output"
BASE_MODEL_PATH="${WORKING_DIR}/models"
BASE_DATA_PATH="${WORKING_DIR}/data"

# Timestamp and experiment name
TIME_STR=599702a5
echo "TIME_STR: ${TIME_STR}"

EXP_NAME="${CUR_MODEL}_${PROJECT_NAME}_${TIME_STR}"

# Output directories
CHECKPOINT_DIR="${BASE_OUTPUT_DIR}/checkpoints/${EXP_NAME}"
LOG_DIR="${BASE_OUTPUT_DIR}/logs/${EXP_NAME}"
WANDB_DIR="${LOG_DIR}/wandb"
export WANDB_DIR="${LOG_DIR}/wandb"
mkdir -p "$WANDB_DIR"

ROLLOUT_DIR="${LOG_DIR}/rollout"
mkdir -p "$ROLLOUT_DIR"

VAL_DIR="${LOG_DIR}/validation"
mkdir -p "$VAL_DIR"

# Data paths
TRAIN_FILE="${BASE_DATA_PATH}/dapo-math-17k-processed.parquet"
TEST_FILE="['${BASE_DATA_PATH}/aime-2024.parquet','${BASE_DATA_PATH}/aime-2025.parquet','${BASE_DATA_PATH}/amc23.parquet','${BASE_DATA_PATH}/math500.parquet','${BASE_DATA_PATH}/minervamath.parquet','${BASE_DATA_PATH}/olympiad.parquet']"
MODEL_PATH="${BASE_MODEL_PATH}/${CUR_MODEL}"

save_freq=66
save_times_per_epoch=0

test_freq=66
val_batch_size=256
adv_estimator='grpo'

# Length control
max_prompt_length=2048
max_response_length=4096

# Calculate maximum token length (for dynamic batch size)
ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length))

# Batch Sizes
train_prompt_bsz=256
ppo_mini_batch_size=128
ppo_micro_batch_size_per_gpu=16
log_prob_micro_batch_size_per_gpu=16

# Dynamic batch size configuration
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1

# Rollout parameters
n_resp_per_prompt=16
gen_tp=4
gpu_memory_utilization=0.6

# Training parameters
optim_lr=1e-6
lr_warmup_steps=10
use_kl_loss=False
kl_loss_coef=0.001
kl_loss_type='low_var_kl'
entropy_coeff=0.0

#################### CoBA-RL Allocator Configuration ####################
# CoBA-RL: Constraint-based Budget Allocation for Reinforcement Learning
# Uses exploit2explore strategy to dynamically allocate rollout times for samples
allocator_type="beta"
allocator_n_low=2                                    # Minimum rollout times
allocator_n_up=128                                   # Maximum rollout times
allocator_sliding_window_size=1                      # Sliding window size
allocator_beta_params_sum=11                         # Beta parameter sum (alpha + beta)
#################################################################

# Determine HEAD_ADDR for multi-node setup
if [ -n "${AFO_ENV_CLUSTER_SPEC:-}" ]; then
    HEAD_ADDR=$(python3 -c "import os, json, socket; spec = json.loads(os.environ['AFO_ENV_CLUSTER_SPEC']); role = spec['role']; master = spec[role][0]; master_addr, _ = master.split(':'); print(socket.gethostbyname(master_addr))")
    echo "====[DEBUG] Parsed head IP from AFO_ENV_CLUSTER_SPEC: $HEAD_ADDR ===="
    export HEAD_ADDR
elif [ -n "${HOPE_HOSTS:-}" ]; then
    IFS=',' read -ra HOST_ARRAY <<< "$HOPE_HOSTS"
    HEAD_ADDR="${HOST_ARRAY[0]}"
    echo "====[DEBUG] Parsed head IP from HOPE_HOSTS: $HEAD_ADDR ===="
    export HEAD_ADDR
else
    HEAD_ADDR=$(hostname -i)
    echo "====[DEBUG] Parsed head IP from hostname -i: $HEAD_ADDR ===="
    export HEAD_ADDR
fi

# Setup directories
LOG_FILE="${LOG_DIR}/train_logs/output_${TIME_STR}.log"
mkdir -p $(dirname $LOG_FILE)

# Ray port configuration
METRICS_EXPORT_PORT=20541
DASHBOARD_AGENT_HTTP_PORT=52365
DASHBOARD_AGENT_GRPC_PORT=53589
RUNTIME_ENV_AGENT_PORT=48869
MIN_WORKER_PORT=10002
MAX_WORKER_PORT=12001
PET_MASTER_PORT=8278
DASHBOARD_PORT=8265
REDIS_SHARD_PORTS="20007,20008,20009,20010,20011,20012,20013,20014,20015,20016"

# Get node rank from AFO_ENV_CLUSTER_SPEC or hope_info.py
CURRPWD=$(pwd)
if [ -n "${AFO_ENV_CLUSTER_SPEC:-}" ]; then
    # Parse rank from AFO_ENV_CLUSTER_SPEC index field
    rank=$(python3 -c "import os, json; spec = json.loads(os.environ['AFO_ENV_CLUSTER_SPEC']); print(spec.get('index', 0))")
    main="$HEAD_ADDR"
    echo "====[DEBUG] Parsed rank from AFO_ENV_CLUSTER_SPEC: rank=$rank, main=$main ===="
elif [ -f "${CURRPWD}/connection/hope_info.py" ]; then
    main_output=$(python3 ${CURRPWD}/connection/hope_info.py) || {
        echo "Warning: hope_info.py execution failed, using default values"
        main_output="localhost 0"
    }
    echo "main_output: $main_output"
    IFS=' ' read -r main rank <<< "$main_output"
else
    main_output="localhost 0"
    echo "main_output: $main_output"
    IFS=' ' read -r main rank <<< "$main_output"
fi
echo "main: $main"
echo "rank: $rank"

# Training command function
run_training() {
    python3 -m recipe.coba_rl.main_coba_rl \
        --config-path="${CONFIG_PATH}" \
        --config-name='coba_rl' \
        algorithm.adv_estimator=${adv_estimator} \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        data.train_batch_size=${train_prompt_bsz} \
        data.val_batch_size=${val_batch_size} \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=${optim_lr} \
        actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
        actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
        actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
        actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.multi_stage_wake_up=True \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.over_sample_rate=0.1 \
        actor_rollout_ref.rollout.mode=sync \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name=${PROJECT_NAME} \
        trainer.experiment_name="${EXP_NAME}" \
        trainer.default_local_dir="${CHECKPOINT_DIR}" \
        trainer.n_gpus_per_node=${n_gpus_per_node} \
        trainer.nnodes=${N_NODE} \
        trainer.save_freq=${save_freq} \
        trainer.save_times_per_epoch=${save_times_per_epoch} \
        trainer.val_before_train=True \
        trainer.total_epochs=${total_epochs} \
        trainer.wandb_dir=${WANDB_DIR} \
        trainer.test_freq=${test_freq} \
        trainer.auto_merge_checkpoints=True \
        trainer.rollout_data_dir=${ROLLOUT_DIR} \
        trainer.validation_data_dir="${VAL_DIR}" \
        actor_rollout_ref.rollout.val_kwargs.n=16 \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        +trainer.allocator_type=${allocator_type} \
        +trainer.allocator_config.n_low=${allocator_n_low} \
        +trainer.allocator_config.n_up=${allocator_n_up} \
        +trainer.allocator_config.sliding_window_size=${allocator_sliding_window_size} \
        +trainer.allocator_config.beta_params_sum=${allocator_beta_params_sum} \
        actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 "${@:1}"
}

echo "current rank: $rank"
# Multi-node or single-node execution
if [ "$N_NODE" -eq 1 ]; then
    echo "====[INFO] Running in single-node mode ===="
    run_training "$@" 2>&1 | tee "${LOG_FILE}"
else
    echo "====[INFO] Running in multi-node mode with $N_NODE nodes ===="

    if [ "$rank" -eq 0 ]; then
        # Head node: start Ray head and submit job
        ray start --head \
            --port=$PET_MASTER_PORT \
            --dashboard-host='0.0.0.0' \
            --metrics-export-port=$METRICS_EXPORT_PORT \
            --dashboard-agent-grpc-port=$DASHBOARD_AGENT_GRPC_PORT \
            --dashboard-agent-listen-port=$DASHBOARD_AGENT_HTTP_PORT \
            --runtime-env-agent-port=$RUNTIME_ENV_AGENT_PORT \
            --dashboard-port=$DASHBOARD_PORT \
            --redis-shard-ports=$REDIS_SHARD_PORTS \
            --min-worker-port=$MIN_WORKER_PORT \
            --max-worker-port=$MAX_WORKER_PORT

        # Wait for worker nodes to join
        sleep 60

        echo "Waiting for all nodes to join Ray cluster..."
        start_time=$(date +%s)
        timeout=600  # 10 minutes timeout

        while true; do
            current_time=$(date +%s)
            elapsed=$((current_time - start_time))

            active_nodes=$(ray status | sed -n '/Active:/,/Pending:/p' | grep "1 node_" | wc -l)
            echo "Current active nodes: $active_nodes / $N_NODE"

            if [ "$active_nodes" -ge "$N_NODE" ]; then
                echo "All nodes have joined the cluster!"
                break
            fi

            if [ "$elapsed" -ge "$timeout" ]; then
                echo "Error: Timeout waiting for nodes to join cluster (10 minutes)"
                echo "Expected nodes: $N_NODE, Actual nodes: $active_nodes"
                exit 1
            fi

            echo "Waiting for more nodes... (${elapsed}s / ${timeout}s)"
            sleep 10
        done

        # Submit training job
        ray job submit --verbose -- \
            python3 -m recipe.coba_rl.main_coba_rl \
            --config-path="${CONFIG_PATH}" \
            --config-name='coba_rl' \
            algorithm.adv_estimator=${adv_estimator} \
            data.train_files="${TRAIN_FILE}" \
            data.val_files="${TEST_FILE}" \
            data.train_batch_size=${train_prompt_bsz} \
            data.val_batch_size=${val_batch_size} \
            data.max_prompt_length=${max_prompt_length} \
            data.max_response_length=${max_response_length} \
            data.filter_overlong_prompts=True \
            data.truncation='error' \
            data.return_raw_chat=True \
            actor_rollout_ref.model.path="${MODEL_PATH}" \
            actor_rollout_ref.actor.optim.lr=${optim_lr} \
            actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
            actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
            actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
            actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
            actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
            actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
            actor_rollout_ref.actor.clip_ratio_low=0.2 \
            actor_rollout_ref.actor.clip_ratio_high=0.28 \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
            actor_rollout_ref.rollout.name=sglang \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
            actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
            actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
            actor_rollout_ref.rollout.enable_chunked_prefill=True \
            actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
            actor_rollout_ref.rollout.multi_stage_wake_up=True \
            actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
            actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            actor_rollout_ref.rollout.over_sample_rate=0.1 \
            actor_rollout_ref.rollout.mode=sync \
            algorithm.use_kl_in_reward=False \
            trainer.critic_warmup=0 \
            trainer.logger='["console","wandb"]' \
            trainer.project_name=${PROJECT_NAME} \
            trainer.experiment_name="${EXP_NAME}" \
            trainer.default_local_dir="${CHECKPOINT_DIR}" \
            trainer.n_gpus_per_node=${n_gpus_per_node} \
            trainer.nnodes=${N_NODE} \
            trainer.save_freq=${save_freq} \
            trainer.save_times_per_epoch=${save_times_per_epoch} \
            trainer.val_before_train=True \
            trainer.total_epochs=${total_epochs} \
            trainer.wandb_dir=${WANDB_DIR} \
            trainer.test_freq=${test_freq} \
            trainer.auto_merge_checkpoints=True \
            trainer.rollout_data_dir=${ROLLOUT_DIR} \
            trainer.validation_data_dir="${VAL_DIR}" \
            actor_rollout_ref.rollout.val_kwargs.n=16 \
            actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
            actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=True \
            +trainer.allocator_type=${allocator_type} \
            +trainer.allocator_config.n_low=${allocator_n_low} \
            +trainer.allocator_config.n_up=${allocator_n_up} \
            +trainer.allocator_config.sliding_window_size=${allocator_sliding_window_size} \
            +trainer.allocator_config.beta_params_sum=${allocator_beta_params_sum} \
            actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 "${@:1}" | tee "${LOG_FILE}"

        # Signal completion to worker nodes
        mkdir -p ${LOG_DIR}/connection/log
        touch ${LOG_DIR}/connection/log/main_done_${main}.txt
        sleep 15

    else
        # Worker node: join Ray cluster
        sleep 30
        ray start --address="$main:$PET_MASTER_PORT" \
            --metrics-export-port=$METRICS_EXPORT_PORT \
            --dashboard-agent-grpc-port=$DASHBOARD_AGENT_GRPC_PORT \
            --runtime-env-agent-port=$RUNTIME_ENV_AGENT_PORT \
            --dashboard-agent-listen-port=$DASHBOARD_AGENT_HTTP_PORT \
            --dashboard-port=$DASHBOARD_PORT \
            --min-worker-port=$MIN_WORKER_PORT \
            --max-worker-port=$MAX_WORKER_PORT

        echo "Ray worker joined master at $main:$PET_MASTER_PORT"

        # Wait for head node to finish
        while [ ! -f ${LOG_DIR}/connection/log/main_done_${main}.txt ]; do
            echo "Waiting for main node to finish..."
            sleep 600
        done
    fi
fi
