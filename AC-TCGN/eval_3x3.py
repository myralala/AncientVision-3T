import json, os, argparse
from typing import List, Dict, Any
from types import MethodType

import torch
import torch.nn.functional as F
from tqdm import tqdm
import Levenshtein as Lev

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

LABELS = ["军事", "儀器", "冠服", "樂器"]

# =======================
# Metrics
# =======================

def calculate_1_ned(pred: str, gt: str) -> float:
    pred = pred.replace(" ", "").replace("\n", "")
    gt   = gt.replace(" ", "").replace("\n", "")
    if len(gt) == 0:
        return 0.0 if len(pred) > 0 else 1.0
    dist = Lev.distance(pred, gt)
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 1.0
    return 1.0 - (dist / max_len)

def lcs_len(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]

def rouge_l(pred: str, gt: str) -> float:
    pred, gt = pred.replace(" ", ""), gt.replace(" ", "")
    if not gt:
        return 1.0 if not pred else 0.0
    lcs = lcs_len(pred, gt)
    p, r = lcs / len(pred), lcs / len(gt)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)

def extract_label_from_text(text: str) -> str:
    for lb in LABELS:
        if lb in text:
            return lb
    return text.strip()

def score(task: str, preds, gts):
    if task == "OCR":
        return sum(calculate_1_ned(p,g) for p,g in zip(preds,gts)) / len(gts)
    if task == "IC":
        return sum(extract_label_from_text(p)==extract_label_from_text(g) for p,g in zip(preds,gts)) / len(gts)
    return sum(rouge_l(p,g) for p,g in zip(preds,gts)) / len(gts)

# =======================
# Data utils
# =======================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_messages(item):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": item["images"][0]},
            {"type": "text", "text": item.get("instruction","") + ("\n"+item.get("input","") if item.get("input") else "")}
        ]
    }]

def prepare_batch(processor, batch):
    msgs = [build_messages(it) for it in batch]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    imgs, vids = process_vision_info(msgs)
    return processor(text=texts, images=imgs, videos=vids, padding=True, return_tensors="pt")

# =======================
# Model helpers
# =======================

def locate_layers(model):
    while hasattr(model, "module"):
        model = model.module
    for k in ["model","language_model","llm","decoder"]:
        obj = getattr(model, k, None)
        if obj and hasattr(obj, "layers"):
            return obj.layers
    raise RuntimeError("Cannot find transformer layers")

class MaskManager:
    def __init__(self, model):
        self.model = model
        self.layers = locate_layers(model)
        self.orig = {i: l.mlp.forward for i,l in enumerate(self.layers)}

    def reset(self):
        for i,l in enumerate(self.layers):
            l.mlp.forward = self.orig[i]

    def apply_mask(self, masks: List[torch.Tensor], alpha: float):
        self.reset()
        if masks is None:
            return

        def factory(mask_idx, alpha_val):
            def forward(self, x):
                if hasattr(self, "gate_up_proj"):
                    gate_up,_ = self.gate_up_proj(x)
                    g,u = gate_up.chunk(2, dim=-1)
                    act = F.silu(g)
                else:
                    g = self.gate_proj(x)
                    u = self.up_proj(x)
                    act = F.silu(g)

                if mask_idx.numel() > 0:
                    scale = torch.ones(act.size(-1), device=act.device, dtype=act.dtype)
                    scale[mask_idx] = alpha_val
                    act = act * scale

                return self.down_proj(act * u)
            return forward

        device = next(self.model.parameters()).device
        for i,mask in enumerate(masks):
            if mask is None or i >= len(self.layers): continue
            self.layers[i].mlp.forward = MethodType(factory(mask.to(device), alpha), self.layers[i].mlp)

# =======================
# Inference
# =======================

def generate(model, processor, data, bs, max_new):
    device = next(model.parameters()).device
    preds,gts = [],[]
    for i in range(0,len(data),bs):
        batch = data[i:i+bs]
        inputs = prepare_batch(processor,batch)
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new)
        trim = [o[inputs["input_ids"].size(1):] for o in out]
        texts = processor.batch_decode(trim, skip_special_tokens=True)
        for t,it in zip(texts,batch):
            gt = it["output"]["content"] if isinstance(it["output"],dict) else it["output"]
            preds.append(t.strip())
            gts.append(gt.strip())
    return preds,gts

# =======================
# Main
# =======================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data-1/Qwen3-VL-8B-Instruct")
    ap.add_argument("--mask_ocr", required=True)
    ap.add_argument("--mask_ic", required=True)
    ap.add_argument("--mask_iu", required=True)
    ap.add_argument("--data_ocr", default="./data/OCR-Eval.json")
    ap.add_argument("--data_ic", default="./data/IC-Eval.json")
    ap.add_argument("--data_iu", default="./data/IU-Eval.json")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--alpha", type=float, default=0.1, help="Activation scaling factor")
    args = ap.parse_args()

    dtype = {"fp16":torch.float16,"bf16":torch.bfloat16,"fp32":torch.float32}[args.dtype]

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer.padding_side="left"
    processor.tokenizer.pad_token=processor.tokenizer.eos_token

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    ).eval()

    manager = MaskManager(model)

    masks = {
        "Baseline": None,
        "Suppress_OCR": torch.load(args.mask_ocr, map_location="cpu")[0],
        "Suppress_IC":  torch.load(args.mask_ic,  map_location="cpu")[0],
        "Suppress_IU":  torch.load(args.mask_iu,  map_location="cpu")[0],
    }

    datasets = {
        "OCR": (load_json(args.data_ocr), 512),
        "IC":  (load_json(args.data_ic), 32),
        "IU":  (load_json(args.data_iu), 512),
    }

    print(f"\nAlpha = {args.alpha}")
    print("="*45)

    for mname,mask in masks.items():
        print(f"\n[{mname}]")
        manager.apply_mask(mask, args.alpha)
        for task,(data,max_t) in datasets.items():
            p,g = generate(model,processor,data,args.batch_size,max_t)
            s = score(task,p,g)
            print(f"  {task}: {s:.4f}")

if __name__ == "__main__":
    main()
