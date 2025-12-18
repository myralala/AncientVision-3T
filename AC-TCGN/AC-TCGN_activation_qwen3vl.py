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
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    lcs = lcs_len(pred, gt)
    return lcs / len(gt)

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
    # Apply chat template
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

def get_mlp_up_proj_params(model):
    params = []
    layers = locate_transformer_layers(model)

    if layers is None:
        for name, p in model.named_parameters():
            if name.endswith("mlp.up_proj.weight"):
                mm = re.search(r"(layers|h)\.(\d+)\.", name)
                if mm:
                    params.append((int(mm.group(2)), p))
        if not params:
            raise RuntimeError("Cannot locate transformer layers for this model (and no mlp.up_proj.weight by name).")
        params.sort(key=lambda x: x[0])
        return params

    for i, block in enumerate(layers):
        mlp = getattr(block, "mlp", None)
        if mlp is None:
            continue
        up = getattr(mlp, "up_proj", None)
        if up is None or not hasattr(up, "weight"):
            continue
        params.append((i, up.weight))

    if not params:
        for name, p in model.named_parameters():
            if name.endswith("mlp.up_proj.weight"):
                mm = re.search(r"(layers|h)\.(\d+)\.", name)
                if mm:
                    params.append((int(mm.group(2)), p))
        params.sort(key=lambda x: x[0])

    if not params:
        raise RuntimeError("No mlp.up_proj.weight found.")
    return params

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True, choices=["OCR","IC","IU"])
    ap.add_argument("--Activationain_path", type=str, default=None)
    ap.add_argument("--model_path", type=str, default="/data-1/Qwen3-VL-8B-Instruct")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16","bf16","fp32"])
    ap.add_argument("--out_matrix", type=str, default="A_tr.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()


    if args.train_path is None:
        if args.task == "OCR":
            args.train_path = "./data/OCR-Activation.json"
        elif args.task == "IC":
            args.train_path = "./data/IC-Activation.json"
        else:
            args.train_path = "./data/IU-Activation.json"

    torch.manual_seed(args.seed)

    # logging
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"AC-TCGN_activation_{args.task}_{ts}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_name, encoding="utf-8")])
    logger = logging.getLogger(__name__)
    logger.info(f"task={args.task} train={args.train_path} model={args.model_path}")

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
    model.train()  

    data = load_json(args.train_path)
    if args.max_samples and args.max_samples > 0:
        data = data[:args.max_samples]

    up_params = get_mlp_up_proj_params(model)
    num_layers = max(i for i,_ in up_params) + 1

    inter_size = up_params[0][1].shape[0]
    out_data = [[0.0]*inter_size for _ in range(num_layers)]

    def iter_batches(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i:i+bs]

    device = next(model.parameters()).device
    logger.info(f"device={device} layers={num_layers} inter={inter_size} samples={len(data)}")

    for batch_items in tqdm(list(iter_batches(data, args.batch_size)), desc="activation"):
        inputs, _ = prepare_batch(processor, batch_items)
        inputs = {k: v.to(device) for k,v in inputs.items()}


        gts = [it.get("output","") for it in batch_items]
        norm_gts = []
        for gt in gts:
            if isinstance(gt, dict):
                norm_gts.append(gt.get("content",""))
            else:
                norm_gts.append(str(gt))
        label_ids = tokenize_labels(processor, norm_gts).to(device)
        input_ids = inputs["input_ids"]
        bsz, seq_len = input_ids.shape
        lbl = torch.full_like(input_ids, fill_value=-100)
        for bi in range(bsz):
            l = label_ids.shape[1]
            take = min(l, seq_len)
            lbl[bi, seq_len-take:seq_len] = label_ids[bi, l-take:l]

        outputs = model(**inputs, labels=lbl)
        loss = outputs.loss
        model.zero_grad(set_to_none=True)
        loss.backward()

        for layer_idx, p in up_params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            score = torch.sum(g, dim=1).abs().float().cpu().tolist()
            row = out_data[layer_idx]
            out_data[layer_idx] = [a+b for a,b in zip(score, row)]

    os.makedirs(os.path.dirname(args.out_matrix) or ".", exist_ok=True)
    with open(args.out_matrix, "w", encoding="utf-8") as f:
        json.dump(out_data, f)
    logger.info(f"saved matrix -> {args.out_matrix}")

if __name__ == "__main__":
    main()
