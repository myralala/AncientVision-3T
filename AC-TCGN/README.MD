# AC-TCGN 

This repository contains the implementation of **AC-TCGN**, a tool for identifying and analyzing ancient Chinese document image-text task neurons neurons in Vision-Language Models (e.g., Qwen3-VL, Xunzi-Qwen2-VL).

## 1. Installation

```bash
pip install -r requirements.txt
# Or manually install core dependencies
pip install torch transformers qwen-vl-utils accelerate
```

## 2. Usage Steps

Step 1: Gradient-based Importance Estimation
Compute the identification scores for intermediate neurons using Target-Conditioned Gradients. Instead of simple activation counting, this step calculates the gradients of the task loss $L$ with respect to the up-projection parameters $W_{up}$ to quantify the causal contribution of each neuron.

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


Baseline Evaluation: Test the model's intrinsic capabilities on AncientVision-3T without intervention.

Evaluate the causal effect of the identified neurons by applying strong inhibition and measuring the performance degradation.
```bash
python AC-TCGN_eval_qwen3vl.py \
    --task OCR \
    --model_path /path/to/Qwen3-VL-Instruct
```

**Intervention Evaluation:**
Load the mask file to inhibit ancient Chinese document image-text task neurons and evaluate performance.

```bash
python AC-TCGN_eval_qwen3vl.py \
    --task OCR \
    --model_path /path/to/Qwen3-VL-Instruct \
    --alpha 0.1 \
    --activation_mask ./masks/OCR_mask.pt
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
    --alpha 0.1 \
    --dtype bf16
```
