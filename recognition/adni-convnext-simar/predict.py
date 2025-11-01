import argparse
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

from modules import build_model

IDX_TO_CLASS = {0: "AD", 1: "CN"}


def load_image(path: Path, img_size: int):
    tform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    with Image.open(path) as img:
        img = img.convert("RGB")
    return tform(img).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--images", type=str, nargs="+", required=True)
    ap.add_argument("--model", type=str, default="convnext_tiny")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=2, pretrained=False)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()

    for img_path in args.images:
        x = load_image(Path(img_path), args.img_size).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        pred_idx = int(torch.argmax(logits, dim=1).item())
        print(f"{img_path}: {IDX_TO_CLASS[pred_idx]} | probs={probs}")


if __name__ == "__main__":
    main()
