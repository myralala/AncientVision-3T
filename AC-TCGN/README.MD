# AC-TCGN Usage Guide

This is a tool for task-specific neuron analysis and intervention on Qwen-VL models.

## 1. Installation

```bash
pip install -r requirements.txt
# Or manually install core dependencies
pip install torch transformers qwen-vl-utils accelerate
```

## 2. Usage Steps

### Step 1: Activation Tracing
Calculate and save the neuron importance matrix for a specific task.

```bash
python AC-TCGN_activation_qwen3vl.py \
    --task OCR \
    --model_path /path/to/Qwen3-VL-Instruct \
    --out_matrix ./results/OCR_tr.json \
    --batch_size 4
```
* **Supported Tasks (`--task`)**: `OCR`, `IC`, `IU`
* **Input**: Defaults to reading `./data/{task}-Activation.json`
* **Output**: Generates an importance matrix in JSON format

### Step 2: Mask Generation (Identify)
Filter the top `k%` neurons based on the importance matrix and generate a mask file.

```bash
python AC-TCGN_identify_qwen3vl.py \
    --matrix ./results/OCR_tr.json \
    --top_ratio 0.05 \
    --out_mask ./masks/OCR_mask.pt
```
* **`--top_ratio`**: The ratio of neurons to filter (e.g., 0.05 represents the top 5%)
* **Output**: Generates a mask file in `.pt` format

### Step 3: Evaluation & Intervention (Single Task)

**Baseline Evaluation:**
Test model performance without any intervention.

```bash
python AC-TCGN_eval_qwen3vl.py \
    --task OCR \
    --model_path /path/to/Qwen3-VL-Instruct
```

**Intervention Evaluation:**
Load the mask file to inhibit specific neurons and evaluate performance.

```bash
python AC-TCGN_eval_qwen3vl.py \
    --task OCR \
    --model_path /path/to/Qwen3-VL-Instruct \
    --activation_mask ./masks/OCR_mask_top5.pt
```

### Step 4: 3x3 Cross-Task Evaluation
If you want to run cross-task evaluation across OCR, IC, and IU tasks simultaneously using their respective masks:

```bash
python eval_3x3.py \
    --model_path /data-1/Qwen3-VL-8B-Instruct \
    --mask_ocr ./masks/ocr_mask.pt \
    --mask_ic ./masks/ic_mask.pt \
    --mask_iu ./masks/iu_mask.pt \
    --data_ocr ./data/OCR-Eval.json \
    --data_ic ./data/IC-Eval.json \
    --data_iu ./data/IU-Eval.json \
    --batch_size 4 \
    --dtype bf16
```