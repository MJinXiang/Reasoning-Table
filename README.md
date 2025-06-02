# Reasoning-Table: Exploring Reinforcement Learning for Table Reasoning

<p align="center">
  <a href=""> 🏠 Homepage</a> |
  <a href=""> 📜 Paper</a> | 
  <a href="https://huggingface.co/datasets/TableQAKit/Reasoning-Table"> 🤗 Dataset</a> | 
  <a href="## 🔍 Installation"> 🚀 Installation</a> 
</p>

## 🔥 News


- **[2025.06.03]** 📑 Our paper is now available on [arXiv]().
- **[2025.06.02]** 🎉 We have released our [Reasoning-Table](https://huggingface.co/datasets/TableQAKit/Reasoning-Table) Dataset on Hugging Face!
- **[2025.06.02]** 🎉 We have released the code of Reasoning-Table.


## 👋 Overview

![DataPipeline](assets/main.png)


## 📊 Quick access Reasoning-Table Dataset

We release the full dataset used in the **Reasoning-Table** project for table reasoning tasks. The dataset is hosted on Hugging Face Datasets for easy access and download.

🔗 **Dataset Link**: [TableQAKit/Reasoning-Table](https://huggingface.co/datasets/TableQAKit/Reasoning-Table)

You can download all files programmatically using the `huggingface_hub` library as shown below:

```python
from huggingface_hub import hf_hub_download, login, list_repo_files
import os

repo_id = "TableQAKit/Reasoning-Table"

# Local download directory
download_dir = "you download path"
os.makedirs(download_dir, exist_ok=True)

# List and download all files from the dataset
all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")

for file in all_files:
    print(f"Downloading: {file}")
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=file,
        repo_type="dataset",
        local_dir=download_dir,
        local_dir_use_symlinks=False
    )
    print(f"Saved to: {file_path}")
```

## 🔍 Installation
Please install torch, vllm and ray according to your own environment configuration. We provide a configuration example adapted from TinyZero in the following:
```
pip install -r requirements.txt
```

```
# install torch
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip install vllm==0.6.3
pip install ray
```

Please further install the verl in the current project and flash attention.
```
# verl
pip install -e .

# flash attention 2
pip install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```



## 🧪 Training
For GRPO and PPO training, we provide configuration scripts for different model sizes. You can select the appropriate script based on your hardware resources:

```bash
# For Qwen2.5-3B-Instruct model training
bash train_grpo_3b.sh

# For Qwen2.5-7B-Instruct model training (requires more GPU memory)
bash train_grpo_7b.sh
```

Key parameters you may want to customize:
- `BASE_MODEL`: Path to the base model (e.g., `"/Qwen/Qwen2.5-3B-Instruct"`)
- `DATA_DIR`: Path to your training dataset (e.g., `data/wikitq`)
- `EXPERIMENT_NAME`: Name for your training experiment (used for logging and checkpoints)
- `trainer.n_gpus_per_node`: Number of GPUs to use per node (adjust based on hardware)
- `trainer.total_epochs`: Number of training epochs

The training scripts use [veRL](https://github.com/volcengine/verl) and support both GRPO (Generalized Return Policy Optimization) and standard PPO algorithms for reinforcement learning.

## 🔍 Evaluation
We provide a powerful and easy-to-use evaluation script for table reasoning tasks. It supports multiple datasets and evaluation modes, making it ideal for benchmarking various models across different reasoning challenges.

To evaluate a trained model on a specific benchmark:

```bash
bash evaluation/test.sh
```

The `test.sh` script supports the following features:

- **Multiple Tasks**: Configurable for various benchmarks like WikiTableQuestions (`wikitq`), ToTTo (`totto`), and others via the `TASK_NAME` parameter.
- **Evaluation Modes**:
  - `standard`: Exact match evaluation (default)
  - `llm`: Uses LLM judge to evaluate responses (helpful for more complex reasoning tasks)
  - `combined`: Uses both exact match and LLM-based evaluation

- **Model Configuration**:
  - `MODEL_PATH`: Path to your fine-tuned model for inference
  - `EVAL_MODEL_PATH`: Path to evaluation model (when using LLM-based evaluation)
  - `MODEL_SIZE`: Size of your model (e.g., `3b`, `7b`)
  - `TRAIN_TYPE`: Training method (e.g., `grpo`, `ppo`, `sft`, `base`)

- **Hardware Configuration**:
  - `TENSOR_PARALLEL_SIZE`: Set based on your GPU configuration
  - `BATCH_SIZE`: Adjust based on available memory
  - `LLM_EVAL_BATCH_SIZE`: Batch size for LLM-based evaluation
  
For detailed usage instructions, parameter settings, and example scripts, please refer to the detailed evaluation [README](./evaluation/README.md).

## Acknowledge
Our code is built upon [veRL](https://github.com/volcengine/verl) and [TinyZero](https://github.com/Jiayi-Pan/TinyZero).

## 🖊️ Citation

