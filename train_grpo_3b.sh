export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

export GPUS_PER_NODE=${MLP_WORKER_GPU:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}}
export NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}
export HYDRA_FULL_ERROR=1

echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "NNODES: $NNODES nodes"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

export BASE_MODEL="/Qwen/Qwen2.5-3B-Instruct"
export DATA_DIR="data/wikitq"
export ROLLOUT_TP_SIZE=1
export WANDB_API_KEY=


export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ATTENTION_BACKEND=XFORMERS

export EXPERIMENT_NAME=train_grpo_wikitq-3b

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    +actor_rollout_ref.rollout.model=tool \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.val_before_train=True \
    +trainer.use_observation_mask=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.project_name=Table-R1 \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=3  2>&1 | tee ${EXPERIMENT_NAME}.log
