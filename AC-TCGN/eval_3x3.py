import json, os, math, argparse, datetime, logging
from typing import List, Dict, Any, Tuple
from types import MethodType
import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm
import Levenshtein as Lev

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

LABELS = ["军事", "儀器", "冠服", "樂器"]

# --- Metric Functions ---

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
    if n == 0 or m == 0:
        return 0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        ai = a[i-1]
        row = dp[i]
        prow = dp[i-1]
        for j in range(1, m+1):
            if ai == b[j-1]:
                row[j] = prow[j-1] + 1
            else:
                up = prow[j]
                left = row[j-1]
                row[j] = up if up >= left else left
    return dp[n][m]

def rouge_l(pred: str, gt: str) -> float:
    pred = pred.replace(" ", "")
    gt   = gt.replace(" ", "")
    n_pred = len(pred)
    n_gt = len(gt)
    if n_gt == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0:
        return 0.0
    lcs = lcs_len(pred, gt)
    precision = lcs / n_pred
    recall = lcs / n_gt
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def extract_label_from_text(text: str) -> str:
    for lb in LABELS:
        if lb in text:
            return lb
    return text.strip()

def score(task: str, preds: List[str], gts: List[str]) -> float:
    if task == "OCR":
        return sum(calculate_1_ned(p, g) for p,g in zip(preds,gts)) / max(len(gts), 1)
    if task == "IC":
        correct = 0
        for p,g in zip(preds,gts):
            pl = extract_label_from_text(p)
            gl = extract_label_from_text(g)
            if pl == gl:
                correct += 1
        return correct / max(len(gts), 1)
    return sum(rouge_l(p, g) for p,g in zip(preds,gts)) / max(len(gts), 1)

# --- Utils ---

def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    img = item["images"][0]
    inst = item.get("instruction","")
    inp = item.get("input","")
    user_content = [
        {"type": "image", "image": img},
        {"type": "text", "text": inst + ("\n" + inp if inp else "")},
    ]
    return [{"role": "user", "content": user_content}]

def prepare_batch(processor, batch_items: List[Dict[str, Any]]):
    messages_list = [build_messages(it) for it in batch_items]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    image_inputs, video_inputs = process_vision_info(messages_list)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs

def _unwrap_model(m):
    while hasattr(m, "module"):
        m = m.module
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m

def locate_transformer_layers(model):
    model = _unwrap_model(model)
    candidates = []
    for attr in ["model", "language_model", "text_model", "llm", "decoder", "visual"]:
        obj = getattr(model, attr, None)
        if obj is not None:
            candidates.append(obj)
            obj2 = getattr(obj, "model", None)
            if obj2 is not None:
                candidates.append(obj2)
    candidates.append(model)

    paths = [
        ("layers",), ("model", "layers"), ("decoder", "layers"),
        ("transformer", "layers"), ("transformer", "h"), ("backbone", "layers"),
    ]

    def get_nested(obj, path):
        for p in path:
            if not hasattr(obj, p): return None
            obj = getattr(obj, p)
        return obj

    for obj in candidates:
        for path in paths:
            layers = get_nested(obj, path)
            if layers is None: continue
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers
    return None

def get_layers(model):
    layers = locate_transformer_layers(model)
    if layers is None:
        raise RuntimeError("Cannot locate transformer layers.")
    return layers


class MaskManager:
    def __init__(self, model):
        self.model = model
        self.layers = get_layers(model)
        self.original_forwards = {}
        # Save original forward methods
        for i, layer in enumerate(self.layers):
            self.original_forwards[i] = layer.mlp.forward

    def reset(self):
        """Restore original forwards"""
        for i, layer in enumerate(self.layers):
            layer.mlp.forward = self.original_forwards[i]
        # Clean cache if needed
        # torch.cuda.empty_cache()

    def apply_mask(self, activation_mask: List[torch.Tensor]):
        """Apply a specific mask (requires reset first)"""
        self.reset() # Ensure clean slate
        
        if activation_mask is None:
            return

        def factory(mask_idx: torch.Tensor):
            def forward_mlp(self, x):
                if hasattr(self, "gate_up_proj"):
                    gate_up, _ = self.gate_up_proj(x)  
                    i = gate_up.size(-1)
                    gate = gate_up[:, :, : i // 2]
                    up   = gate_up[:, :, i // 2:]
                    activation = F.silu(gate)
                else:
                    gate = self.gate_proj(x)
                    up   = self.up_proj(x)
                    activation = F.silu(gate)

                if mask_idx is not None and mask_idx.numel() > 0:
                    mask_tensor = torch.ones(activation.size(-1), device=activation.device, dtype=activation.dtype)
                    mask_tensor[mask_idx] = 0
                    activation = activation * mask_tensor

                x = activation * up
                x = self.down_proj(x)
                return x
            return forward_mlp

        device = next(self.model.parameters()).device
        for i, layer_mask in enumerate(activation_mask):
            if i >= len(self.layers): break
            if layer_mask is None: continue
            
            mlp = self.layers[i].mlp
            mask_on_device = layer_mask.to(device)
            mlp.forward = MethodType(factory(mask_on_device), mlp)

def generate_predictions(model, processor, data, batch_size, max_new_tokens):
    device = next(model.parameters()).device
    preds, gts = [], []

    def iter_batches(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i:i+bs]

    for batch_items in tqdm(list(iter_batches(data, batch_size)), desc="Infer", leave=False):
        inputs = prepare_batch(processor, batch_items)
        inputs = {k: v.to(device) for k,v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        input_len = inputs["input_ids"].shape[1]
        gen_ids_trimmed = [out[input_len:] for out in gen_ids]
        texts = processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)
        
        for t, it in zip(texts, batch_items):
            gt = it.get("output","")
            if isinstance(gt, dict): gt = gt.get("content","")
            else: gt = str(gt)
            preds.append(t.strip())
            gts.append(gt.strip())
    return preds, gts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="/data-1/Qwen3-VL-8B-Instruct")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16","bf16","fp32"])
    
    # 3x3 必须的参数
    ap.add_argument("--data_ocr", type=str, default="./data/OCR-Eval.json")
    ap.add_argument("--data_ic", type=str, default="./data/IC-Eval.json")
    ap.add_argument("--data_iu", type=str, default="./data/IU-Eval.json")

    ap.add_argument("--mask_ocr", type=str, required=True)
    ap.add_argument("--mask_ic", type=str, required=True)
    ap.add_argument("--mask_iu", type=str, required=True)
    
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    print(f"Loading Qwen3VL from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    mask_manager = MaskManager(model)

    # 抑制配置
    masks = {
        "Suppress_OCR": args.mask_ocr,
        "Suppress_IC":  args.mask_ic,
        "Suppress_IU":  args.mask_iu,
        "Baseline": None
    }

    # 数据集
    eval_tasks = {
        "OCR": {"path": args.data_ocr, "max_tokens": 512},
        "IC":  {"path": args.data_ic,  "max_tokens": 32},
        "IU":  {"path": args.data_iu,  "max_tokens": 512}
    }

    # 预加载数据
    print("Loading datasets...")
    loaded_data = {}
    for t, cfg in eval_tasks.items():
        if os.path.exists(cfg["path"]):
            loaded_data[t] = load_json(cfg["path"])
        else:
            print(f"Warning: {cfg['path']} not found.")

    results = {m_name: {} for m_name in masks}

    print("\n=== Starting 3x3 Evaluation ===")
    
    for mask_name, mask_path in masks.items():
        print(f"\n[Configuration]: {mask_name}")
        
        # 加载并应用 Mask
        if mask_path:
            try:
                saved = torch.load(mask_path, map_location="cpu")
                mask_data = saved[0] if isinstance(saved, (list, tuple)) else saved
                mask_manager.apply_mask(mask_data)
                print(f"  Applied mask from {mask_path}")
            except Exception as e:
                print(f"  Error loading mask: {e}")
                continue
        else:
            mask_manager.reset()
            print("  Baseline (Original Model)")

        # 3
        for task, data_list in loaded_data.items():
            if not data_list: continue
            
            p, g = generate_predictions(
                model, processor, data_list, 
                args.batch_size, 
                eval_tasks[task]["max_tokens"]
            )
            s = score(task, p, g)
            results[mask_name][task] = s
            print(f"  -> {task} Score: {s:.5f}")

    print("\n\n" + "="*45)
    print(f"{'Inhibition':<15} | {'OCR':<8} | {'IC':<8} | {'IU':<8}")
    print("-" * 45)
    for m_name in masks:
        row = results[m_name]
        print(f"{m_name:<15} | {row.get('OCR',0):.4f}   | {row.get('IC',0):.4f}   | {row.get('IU',0):.4f}")

if __name__ == "__main__":
    main()