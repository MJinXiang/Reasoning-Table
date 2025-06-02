# 📊 Unified Inference & Evaluation Script for Table Reasoning Tasks

This script automates the **inference and evaluation** pipeline for table-based reasoning tasks.

It supports:
- Multi-GPU tensor parallel inference via vLLM
- Flexible evaluation modes: **standard**, **llm-based**, or **combined**
- Auto-generated file management for results and logs

---

## 🚀 How to Use

### Step 1: Set Parameters
Edit the top section of the script to configure your experiment:

```bash
# ================= Parameter Configuration =================
MODEL_PATH="path/to/your/inference/model"
EVAL_MODEL_PATH="Qwen/Qwen-14B-Chat"  # or local path to evaluator model
TASK_NAME="totto"
TRAIN_TYPE="grpo"  # or: sft, ppo, base
MODEL_SIZE="14b"
TENSOR_PARALLEL_SIZE=2
BATCH_SIZE=128
MAX_TOKENS=4096
EVAL_MODE="standard"  # mode: standard , combined , llm
LLM_EVAL_BATCH_SIZE=50
```

### Step 2: Run the Script
```bash
bash test.sh
```

This will automatically:

1.Launch vLLM-based batch inference

2.Save predictions and logs

3.Run evaluation based on the selected mode

4.Save evaluation results and logs

## 🔍 Evaluation Modes

| Mode       | Description                                                     |
| ---------- | --------------------------------------------------------------- |
| `standard` | Exact-match metric evaluation based on ground-truth answers     |
| `llm`      | Uses an LLM (e.g., GPT-4, Qwen-Chat) to judge generated outputs |
| `combined` | Runs both exact match and LLM-based evaluation                  |


## 📁 Output Structure
The script automatically saves outputs to the following paths:
```
results/
  └── <TASK_NAME>/
        ├── logs/
        │     └── <TASK>_<MODEL_SIZE>_<TRAIN_TYPE>_*.log
        ├── <TASK>_<MODEL_SIZE>_<TRAIN_TYPE>.json              # Prediction
        └── <TASK>_<MODEL_SIZE>_<TRAIN_TYPE>_eval_results.json # Eval result
```

## ✅ Notes
LLM evaluation requires internet access if using models like Qwen/Qwen-14B-Chat or GPT-4 via API.

Tensor parallel size must match your GPU configuration and model attention head count.

You can easily swap in any table benchmark by changing TASK_NAME and corresponding test script.

## 📫 Contact
If you encounter issues or want to contribute support for more datasets or evaluation models, feel free to submit an issue or pull request.