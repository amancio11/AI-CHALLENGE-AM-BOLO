"""
analyze_facade.py  —  Pipeline: YOLOv8 → EfficientNet-B2 → Grad-CAM → report JSON

Uso:
    # Singola immagine (come prima)
    python src/inference/analyze_facade.py --image am_images_test/TEST_AM_YOLO/b.jpg

    # Tutta una cartella → output in runs/inference/<nome_cartella>/
    python src/inference/analyze_facade.py --folder am_images_test/PuliteSara

    # Cartella con output personalizzato
    python src/inference/analyze_facade.py --folder am_images_test/TEST_AM_YOLO --out runs/inference/batch1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms
from ultralytics import YOLO

ROOT            = Path(__file__).resolve().parents[2]
YOLO_PATH       = ROOT / "models" / "weights" / "yolov8_windows_v3_merged_best.pt"
EFFNET_PATH     = ROOT / "models" / "weights" / "efficientnet_best.pt"
MIN_CONF        = 0.50   # non usato direttamente — sostituito da dual-threshold
CONF_LARGE      = 0.30   # soglia conf per box grandi
CONF_SMALL      = 0.70   # soglia conf per box piccoli
LARGE_SIZE      = 300    # lato minore del box (px) sopra cui è considerato "grande"
MIN_CROP_SIZE   = 66
PADDING_PCT     = 0.05
IMAGE_SIZE      = 260
CLASS_NAMES     = ["clean", "dirty"]
COLOR_CLEAN     = (50, 200, 50)
COLOR_DIRTY     = (220, 30, 30)

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_effnet(path: Path, device: torch.device) -> torch.nn.Module:
    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def pad_crop(img: Image.Image, x1, y1, x2, y2):
    w, h = img.size
    px, py = (x2 - x1) * PADDING_PCT, (y2 - y1) * PADDING_PCT
    x1c, y1c = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2c, y2c = int(min(w, x2 + px)), int(min(h, y2 + py))
    x2c, y2c = max(x2c, x1c + 1), max(y2c, y1c + 1)
    return img.crop((x1c, y1c, x2c, y2c)), (x1c, y1c, x2c, y2c)


class GradCAM:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._act = self._grad = None
        layer = model.blocks[-1]
        layer.register_forward_hook(lambda m, i, o: setattr(self, "_act", o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "_grad", go[0].detach()))

    def __call__(self, inp: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        self.model(inp)[0, 1].backward()
        weights = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._act).sum(1)).squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_heatmap(img: Image.Image, cam: np.ndarray, x1, y1, x2, y2, alpha=0.45):
    """Sovrappone una heatmap rossa sul crop (x1,y1,x2,y2) dell'immagine PIL."""
    h, w = y2 - y1, x2 - x1
    cam_r = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h)))
    # colore: rosso = sporco (alto), blu = pulito (basso) — JET semplificato
    r = np.clip(cam_r * 2, 0, 255).astype(np.uint8)
    g = np.zeros_like(r)
    b = np.clip(255 - cam_r * 2, 0, 255).astype(np.uint8)
    hm = Image.fromarray(np.stack([r, g, b], axis=2))
    roi = img.crop((x1, y1, x2, y2))
    blended = Image.blend(roi, hm, alpha)
    img.paste(blended, (x1, y1))


def filter_nested(detections: list, threshold: float = 0.80) -> list:
    """Scarta i box grandi che contengono (>= threshold) un box più piccolo."""
    if len(detections) <= 1:
        return detections
    boxes = [d["bbox"] for d in detections]
    n = len(boxes)
    remove = set()
    for i in range(n):
        for j in range(n):
            if i == j or j in remove:
                continue
            ax1, ay1, ax2, ay2 = boxes[i]
            bx1, by1, bx2, by2 = boxes[j]
            inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
            area_i = (ax2 - ax1) * (ay2 - ay1)
            area_j = (bx2 - bx1) * (by2 - by1)
            if area_i > 0 and area_j > area_i and inter / area_i >= threshold:
                remove.add(j)
    result = [d for k, d in enumerate(detections) if k not in remove]
    if removed := len(detections) - len(result):
        print(f"[~] Rimossi {removed} box annidati")
    return result


def draw_box(draw: ImageDraw.Draw, x1, y1, x2, y2, color, label: str, img_size: tuple):
    thickness = max(2, img_size[0] // 600)
    for t in range(thickness):
        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
    font_size = max(16, img_size[0] // 60)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    bb = draw.textbbox((0, 0), label, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = 3
    ty = max(0, y1 - th - pad * 2)
    draw.rectangle([x1, ty, x1 + tw + pad * 2, ty + th + pad * 2], fill=color)
    draw.text((x1 + pad, ty + pad), label, fill=(255, 255, 255), font=font)


def analyze(image_path: Path, out_img: Path, out_json: Path, min_conf: float, min_size: int,
            conf_large: float = CONF_LARGE, conf_small: float = CONF_SMALL, large_size: int = LARGE_SIZE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[+] Carico YOLOv8:       {YOLO_PATH.name}")
    yolo = YOLO(str(YOLO_PATH))
    print(f"[+] Carico EfficientNet: {EFFNET_PATH.name}")
    effnet = load_effnet(EFFNET_PATH, device)
    gradcam = GradCAM(effnet)

    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    print(f"[+] Immagine: {image_path.name}  ({image.width}x{image.height} px)")

    # Rilevamento finestre — usa soglia bassa per catturare tutto, poi dual-threshold per dimensione
    results = yolo.predict(image, conf=min(conf_large, conf_small), iou=0.45, verbose=False)
    detections = []
    for r in results:
        for box in (r.boxes or []):
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            w_box, h_box = x2 - x1, y2 - y1
            if w_box < min_size or h_box < min_size:
                continue
            conf_box = float(box.conf[0])
            is_large = min(w_box, h_box) >= large_size
            threshold = conf_large if is_large else conf_small
            if conf_box >= threshold:
                detections.append({"conf_yolo": round(conf_box, 4),
                                   "bbox": [x1, y1, x2, y2],
                                   "size_class": "large" if is_large else "small"})

    detections = filter_nested(detections)
    print(f"[+] Finestre rilevate (large≥{conf_large}, small≥{conf_small}, size≥{min_size}px): {len(detections)}")

    draw = ImageDraw.Draw(image)
    report = []

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        crop, (xc1, yc1, xc2, yc2) = pad_crop(image, x1, y1, x2, y2)

        inp = TRANSFORM(crop).unsqueeze(0).to(device)
        inp.requires_grad_(True)
        with torch.set_grad_enabled(True):
            probs = F.softmax(effnet(inp), dim=1)[0]
        pred = int(probs.argmax())
        label = CLASS_NAMES[pred]
        conf = float(probs[pred].detach())

        # Grad-CAM solo per finestre dirty
        if pred == 1:
            cam = gradcam(inp)
            overlay_heatmap(image, cam, xc1, yc1, xc2, yc2)
            draw = ImageDraw.Draw(image)  # ridisegna su immagine modificata

        color = COLOR_CLEAN if pred == 0 else COLOR_DIRTY
        draw_box(draw, x1, y1, x2, y2, color, f"{label} {conf:.0%}", image.size)

        report.append({
            "window_id": i,
            "bbox": [x1, y1, x2, y2],
            "conf_yolo": det["conf_yolo"],
            "label": label,
            "conf": round(conf, 4),
            "dirty_prob": round(float(probs[1].detach()), 4),
            "clean_prob": round(float(probs[0].detach()), 4),
        })
        print(f"    window {i:02d}: {label:<5}  conf={conf:.2%}  bbox={det['bbox']}")

    out_img.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_img, quality=95)
    print(f"\n[+] Immagine annotata: {out_img}")

    summary = {
        "image": image_path.name,
        "total": len(report),
        "clean": sum(1 for r in report if r["label"] == "clean"),
        "dirty": sum(1 for r in report if r["label"] == "dirty"),
        "windows": report,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Report JSON:       {out_json}")
    print(f"\n── Riepilogo: {summary['clean']} clean, {summary['dirty']} dirty "
          f"({summary['dirty']}/{summary['total']} sporche)")


if __name__ == "__main__":
    SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Singola immagine")
    group.add_argument("--folder", type=Path, help="Cartella di immagini")
    parser.add_argument("--conf", type=float, default=MIN_CONF)  # legacy, ignorato se si usano i due sotto
    parser.add_argument("--conf-large", type=float, default=CONF_LARGE, help="Conf minima box grandi (default 0.30)")
    parser.add_argument("--conf-small", type=float, default=CONF_SMALL, help="Conf minima box piccoli (default 0.70)")
    parser.add_argument("--large-size", type=int, default=LARGE_SIZE, help="Lato minimo (px) per box 'grande' (default 300)")
    parser.add_argument("--min-size", type=int, default=MIN_CROP_SIZE)
    parser.add_argument("--out", type=Path, default=None, help="Output dir (solo con --folder) o file (con --image)")
    parser.add_argument("--json", default=None)
    parser.add_argument("--contain-thresh", type=float, default=0.80)
    args = parser.parse_args()

    out_dir = ROOT / "runs" / "inference"

    if args.image:
        img = args.image
        if not img.exists():
            raise FileNotFoundError(img)
        out_img  = args.out if args.out else out_dir / f"{img.stem}_analyzed.jpg"
        out_json = Path(args.json) if args.json else out_dir / f"{img.stem}_report.json"
        analyze(img, out_img, out_json, args.conf, args.min_size,
                args.conf_large, args.conf_small, args.large_size)

    else:
        folder = args.folder
        if not folder.exists():
            raise FileNotFoundError(folder)
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXT)
        if not images:
            raise SystemExit(f"[!] Nessuna immagine trovata in {folder}")
        batch_dir = args.out if args.out else out_dir / folder.name
        batch_dir.mkdir(parents=True, exist_ok=True)
        print(f"[+] {len(images)} immagini in {folder}  →  output in {batch_dir}\n")
        for img in images:
            out_img  = batch_dir / f"{img.stem}_analyzed.jpg"
            out_json = batch_dir / f"{img.stem}_report.json"
            print(f"{'─'*50}")
            analyze(img, out_img, out_json, args.conf, args.min_size,
                    args.conf_large, args.conf_small, args.large_size)
            print()
