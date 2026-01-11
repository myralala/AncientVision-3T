import json, os, math, argparse, datetime, logging
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
import Levenshtein as Lev

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

LABELS = ["军事", "儀器", "冠服", "樂器"]

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


# ROUGE-L
def rouge_l(pred: str, gt: str) -> float:
    """
    计算字符级 ROUGE-L F1 Score
    F1 = 2 * P * R / (P + R)
    """
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

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def extract_label_from_text(text: str) -> str:
    for lb in LABELS:
        if lb in text:
            return lb
    return text.strip()

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
    return inputs, messages_list

def tokenize_labels(processor, labels: List[str]) -> torch.Tensor:
    tok = processor.tokenizer(
        labels,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return tok["input_ids"]


from types import MethodType

def _unwrap_model(m):
    while hasattr(m, "module"):
        m = m.module
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m

def locate_transformer_layers(model):
    model = _unwrap_model(model)
    candidates = []
    for attr in ["model", "language_model", "text_model", "llm", "decoder"]:
        obj = getattr(model, attr, None)
        if obj is not None:
            candidates.append(obj)
            obj2 = getattr(obj, "model", None)
            if obj2 is not None:
                candidates.append(obj2)
    candidates.append(model)

    paths = [
        ("layers",),
        ("model", "layers"),
        ("decoder", "layers"),
        ("transformer", "layers"),
        ("transformer", "h"),
        ("backbone", "layers"),
    ]

    def get_nested(obj, path):
        for p in path:
            if not hasattr(obj, p):
                return None
            obj = getattr(obj, p)
        return obj

    for obj in candidates:
        for path in paths:
            layers = get_nested(obj, path)
            if layers is None:
                continue
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers
    return None

def get_layers(model):
    layers = locate_transformer_layers(model)
    if layers is None:
        raise RuntimeError("Cannot locate transformer layers for this model.")
    return layers

def patch_mlp_forward(model, activation_mask, alpha: float = 0.1):
    """
    activation_mask: list[tensor] length = n_layers, each tensor holds neuron indices (to be scaled)
    alpha: scaling factor for masked neurons (default 0.1). Unmasked neurons keep scale=1.
    """
    layers = get_layers(model)

    def factory(mask_idx: torch.Tensor, alpha_val: float):
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
                mask_tensor = torch.ones(
                    activation.size(-1),
                    device=activation.device,
                    dtype=activation.dtype
                )
                mask_tensor[mask_idx] = alpha_val
                activation = activation * mask_tensor

            x = activation * up
            x = self.down_proj(x)
            return x
        return forward_mlp

    device = next(model.parameters()).device

    for i, layer_mask in enumerate(activation_mask):
        if i >= len(layers):
            break
        mlp = layers[i].mlp
        if layer_mask is None:
            continue
        mlp.forward = MethodType(factory(layer_mask.to(device), alpha), mlp)

def generate_predictions(model, processor, data, batch_size, max_new_tokens):
    device = next(model.parameters()).device
    preds, gts = [], []

    def iter_batches(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i:i+bs]

    for batch_items in tqdm(list(iter_batches(data, batch_size)), desc="eval"):
        inputs, _ = prepare_batch(processor, batch_items)
        inputs = {k: v.to(device) for k,v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        input_len = inputs["input_ids"].shape[1]
        gen_ids_trimmed = [out[input_len:] for out in gen_ids]
        texts = processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)

        for t, it in zip(texts, batch_items):
            gt = it.get("output","")
            if isinstance(gt, dict):
                gt = gt.get("content","")
            else:
                gt = str(gt)
            preds.append(t.strip())
            gts.append(gt.strip())
    return preds, gts

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True, choices=["OCR","IC","IU"])
    ap.add_argument("--eval_path", dest="eval_path", type=str, default=None)
    ap.add_argument("--model_path", type=str, default="/data-1/Qwen3-VL-8B-Instruct")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16","bf16","fp32"])
    ap.add_argument("--activation_mask", type=str, default="", help="torch mask file from identify stage (optional)")
    ap.add_argument("--alpha", type=float, default=0.1, help="Activation scaling factor for masked neurons (default: 0.1)")

    args = ap.parse_args()

    if args.eval_path is None:
        if args.task == "OCR":
            args.eval_path = "./data/OCR-Eval.json"
            default_max = 512
        elif args.task == "IC":
            args.eval_path = "./data/IC-Eval.json"
            default_max = 32
        else:
            args.eval_path = "./data/IU-Eval.json"
            default_max = 512
    else:
        default_max = 64

    if args.max_new_tokens is None:
        args.max_new_tokens = default_max

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

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

    activation_masks = None
    if args.activation_mask:
        saved = torch.load(args.activation_mask, map_location="cpu")
        activation_masks = saved[0]
        patch_mlp_forward(model, activation_masks, alpha=args.alpha)

    data = load_json(args.eval_path)
    preds, gts = generate_predictions(model, processor, data, args.batch_size, args.max_new_tokens)
    s = score(args.task, preds, gts)

    print(
        f"task={args.task}  metric={s:.6f}  "
        f"masked={bool(args.activation_mask)}  alpha={args.alpha if args.activation_mask else 'N/A'}"
    )

if __name__ == "__main__":
    main()
