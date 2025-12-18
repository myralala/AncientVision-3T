import argparse, json, os
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=str, required=True)
    ap.add_argument("--top_ratio", type=float, default=0.05)
    ap.add_argument("--out_mask", type=str, default="activation_mask.pt",)
    args = ap.parse_args()

    with open(args.matrix, "r", encoding="utf-8") as f:
        data = json.load(f)

    matrix = torch.tensor(data)
    layers, number = matrix.shape
    flattened = matrix.view(-1)
    k = int(flattened.numel() * args.top_ratio)
    k = max(k, 1)

    _, top_indices = torch.topk(flattened, k)

    top_indices_2d = torch.stack([top_indices // number, top_indices % number], dim=1)

    output = [[[] for _ in range(layers)]]
    for rc in top_indices_2d:
        l = int(rc[0].item())
        c = int(rc[1].item())
        output[0][l].append(c)

    save_output = []
    save_output.append([torch.tensor(cols, dtype=torch.int64) for cols in output[0]])

    os.makedirs(os.path.dirname(args.out_mask) or ".", exist_ok=True)
    torch.save(save_output, args.out_mask)
    print(f"saved mask -> {args.out_mask}  (layers={layers}, total={k})")

if __name__ == "__main__":
    main()
