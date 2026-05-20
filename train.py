import os
import json
import argparse
import sys
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from torch.optim import AdamW

from groundingdino_loss import GroundingDINOLoss

CONFIG = {
    "batch_size": 4,
    "epochs": 10,
    "lr": 5e-5
}

print("🔧 Device:", "cuda" if torch.cuda.is_available() else "cpu")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, root):
        self.data = [json.loads(x) for x in open(path)]
        self.root = root

        self.t = T.Compose([
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        img = Image.open(os.path.join(self.root, d["file_name"])).convert("RGB")

        w, h = img.size
        img = self.t(img)

        boxes, labels = [], []

        for a in d.get("annotations", []):
            x1, y1, x2, y2 = a["bbox"]
            boxes.append([x1/w, y1/h, x2/w, y2/h])
            labels.append(a["category_name"])

        return img, {
            "boxes": torch.tensor(boxes) if boxes else torch.zeros((0, 4)),
            "labels": labels,
            "caption": d.get("caption", "")
        }, d.get("caption", "")


def collate(b):
    imgs, t, c = zip(*b)
    return torch.stack(imgs), list(t), list(c)


def load_model(device):
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig

    cfg = SLConfig.fromfile("./groundingdino/config/GroundingDINO_SwinB_cfg.py")
    return build_model(cfg).to(device)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="./outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = Dataset(
        os.path.join(args.data_path, "train/annotations/train.jsonl"),
        args.data_path
    )

    dl = DataLoader(
        ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate
    )

    model = load_model(device)
    loss_fn = GroundingDINOLoss().to(device)
    opt = AdamW(model.parameters(), lr=CONFIG["lr"])

    tokenizer = model.tokenizer

    print("🚀 Training...")
    print("=" * 80)

    for e in range(CONFIG["epochs"]):
        pbar = tqdm(enumerate(dl), total=len(dl), desc=f"Epoch {e}")

        loss_history = []
        valid_steps = 0
        skipped_steps = 0

        for i, (imgs, tgts, caps) in pbar:
            imgs = imgs.to(device)

            try:
                tokenized = tokenizer(
                    caps,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)

                out = model(imgs, captions=caps)
                loss_dict = loss_fn(out, tgts, tokenized, tokenizer)
                loss = loss_dict["loss"]

                if torch.isnan(loss):
                    skipped_steps += 1
                    continue

            except Exception as ex:
                skipped_steps += 1
                continue

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_items = {k: v.item() if torch.is_tensor(v) else v 
                         for k, v in loss_dict.items()}
            
            valid_steps += 1

            # Print every 10 steps
            if i % 10 == 0:
                loss_str = " | ".join(
                    [f"{k}={v:.4f}" for k, v in loss_items.items()]
                )
                print(f"[Epoch {e} Step {i:4d}] {loss_str}", flush=True)

            pbar.set_postfix({
                "loss": loss_items.get("loss", 0),
                "valid": valid_steps,
            }, refresh=True)

            loss_history.append(loss_items)

        # Save model
        model_path = os.path.join(args.output_dir, f"model_epoch_{e}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"\n✅ Model saved to {model_path}")

        # Save loss history
        loss_path = os.path.join(args.output_dir, f"loss_epoch_{e}.json")
        with open(loss_path, "w") as f:
            json.dump(loss_history, f, indent=2)

        # Print summary
        if loss_history:
            avg_losses = {
                k: sum(lh[k] for lh in loss_history) / len(loss_history)
                for k in loss_history[0].keys()
            }
            print("\n" + "=" * 80)
            print(f"Epoch {e} Summary:")
            for k, v in avg_losses.items():
                print(f"  {k:15s}: {v:.6f}")
            print("=" * 80 + "\n")


if __name__ == "__main__":
    train()