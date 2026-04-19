"""
app.py  —  Flask web app per AI Challenge Bicchieri Team

Avvio:
    cd AI-CHALLENGE2
    pip install flask openai python-dotenv
    python src/app/app.py

    Apri: http://localhost:5050
"""

import base64
import json
import os
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Carica .env se presente
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from PIL import Image, ImageDraw, ImageFont, ImageOps
sys.path.insert(0, str(ROOT / "src"))

from flask import Flask, jsonify, render_template, request, send_file

from inference.analyze_facade import (
    CONF_LARGE, CONF_SMALL, LARGE_SIZE, MIN_CROP_SIZE, analyze,
)

RUNS_DIR = ROOT / "runs" / "webapp"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = RUNS_DIR / "history.json"
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB


# ── History ────────────────────────────────────────────────────────────────────
def load_history() -> list:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text("utf-8"))
    return []


def save_history(history: list) -> None:
    HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), "utf-8")


# ── Azure GPT-4o fallback (0 finestre rilevate localmente) ─────────────────────
def analyze_with_gpt4o(img_path: Path) -> dict:
    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version="2024-12-01-preview",
        )
        import io
        # Applica EXIF e invia l'immagine già orientata correttamente
        img_pil = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        orig_w, orig_h = img_pil.size

        # Ridimensiona a max 1024px sul lato lungo: GPT-4o è più preciso sulle
        # coordinate quando l'immagine è in un range "umano" di dimensioni
        MAX_SIDE = 1024
        scale = min(MAX_SIDE / orig_w, MAX_SIDE / orig_h, 1.0)
        if scale < 1.0:
            send_w = int(orig_w * scale)
            send_h = int(orig_h * scale)
            img_send = img_pil.resize((send_w, send_h), Image.LANCZOS)
        else:
            img_send = img_pil
            send_w, send_h = orig_w, orig_h

        buf = io.BytesIO()
        img_send.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode()
        mime = "jpeg"

        prompt = (
            "You are an expert building facade cleanliness inspector. "
            "Identify ALL visible windows or glass surfaces and determine if each is clean or dirty. "
            "Return ONLY valid JSON with this exact structure:\n"
            '{"total":<int>,"clean":<int>,"dirty":<int>,'
            '"overall_assessment":"<string>",'
            '"windows":[{"window_id":<int>,"label":"clean or dirty",'
            '"conf":<float 0-1>,"dirty_prob":<float 0-1>,"clean_prob":<float 0-1>,'
            '"bbox":[x1,y1,x2,y2],'
            '"description":"<what you observe>"}]}\n'
            "bbox must use NORMALIZED coordinates in range [0.0, 1.0]: "
            "x1=left/width, y1=top/height, x2=right/width, y2=bottom/height. "
            "Example: a window in the top-left quarter would be [0.0, 0.0, 0.5, 0.5]. "
            "Draw tight bounding boxes — include only the window glass and frame, "
            "NOT the surrounding wall. Crop exactly at the window edges."
        )

        response = client.chat.completions.create(
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/{mime};base64,{b64}",
                        "detail": "high",
                    }},
                ],
            }],
            response_format={"type": "json_object"},
            max_completion_tokens=1500,
        )
        result = json.loads(response.choices[0].message.content)
        result["source"] = "azure_gpt4o"
        # Salva le dimensioni GPT per scalare le bbox al momento del disegno
        result["_gpt_w"] = send_w
        result["_gpt_h"] = send_h
        result["_orig_w"] = orig_w
        result["_orig_h"] = orig_h
        return result

    except KeyError:
        return {
            "total": 0, "clean": 0, "dirty": 0,
            "source": "azure_gpt4o_error",
            "error": "Variabili AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_KEY non configurate.",
            "windows": [],
            "overall_assessment": "Azure GPT-4o non configurato. Imposta AZURE_OPENAI_ENDPOINT e AZURE_OPENAI_KEY.",
        }
    except Exception as exc:
        return {
            "total": 0, "clean": 0, "dirty": 0,
            "source": "azure_gpt4o_error",
            "error": str(exc),
            "windows": [],
            "overall_assessment": f"Analisi Azure GPT-4o fallita: {exc}",
        }


# ── Disegna box GPT-4o sull'immagine ──────────────────────────────────────────
def draw_gpt_boxes(orig_path: Path, out_path: Path, windows: list,
                   gpt_dims: tuple | None = None) -> None:
    """Disegna i bounding box restituiti da GPT-4o.
    gpt_dims=(gpt_w, gpt_h) è la dimensione dell'immagine inviata a GPT;
    le coordinate bbox vengono scalate all'immagine reale.
    """
    img = ImageOps.exif_transpose(Image.open(orig_path)).convert("RGB")
    w, h = img.size
    # Fattore di scala: da spazio-GPT → spazio-immagine reale
    if gpt_dims and gpt_dims[0] and gpt_dims[1]:
        sx = w / gpt_dims[0]
        sy = h / gpt_dims[1]
    else:
        sx = sy = 1.0
    draw = ImageDraw.Draw(img)

    font_size = max(18, w // 55)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    COLOR_CLEAN = (50, 200, 50)
    COLOR_DIRTY = (220, 30, 30)

    for win in windows:
        bbox = win.get("bbox") or win.get("bbox_norm")
        if not bbox or len(bbox) != 4:
            continue

        bx1, by1, bx2, by2 = [float(v) for v in bbox]
        # Coordinate normalizzate [0-1] → pixel reali
        # Se per sbaglio GPT restituisce pixel (valori > 1), normalizza rispetto alle dim GPT
        if max(bx1, by1, bx2, by2) > 1.0:
            gw = gpt_dims[0] if gpt_dims and gpt_dims[0] else w
            gh = gpt_dims[1] if gpt_dims and gpt_dims[1] else h
            bx1, by1, bx2, by2 = bx1/gw, by1/gh, bx2/gw, by2/gh
        # Scala [0-1] → pixel immagine reale
        bx1, by1, bx2, by2 = bx1*w, by1*h, bx2*w, by2*h
        # Clamp ai bordi immagine
        x1 = max(0, min(int(bx1), w-1))
        y1 = max(0, min(int(by1), h-1))
        x2 = max(0, min(int(bx2), w-1))
        y2 = max(0, min(int(by2), h-1))
        if x2 <= x1 or y2 <= y1:
            continue
        label = win.get("label", "?")
        conf  = win.get("conf", 0)
        color = COLOR_CLEAN if label == "clean" else COLOR_DIRTY

        thickness = max(2, w // 600)
        for t in range(thickness):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=color)

        text = f"{label} {conf:.0%}"
        bb   = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        pad = 3
        ty  = max(0, y1 - th - pad*2)
        draw.rectangle([x1, ty, x1+tw+pad*2, ty+th+pad*2], fill=color)
        draw.text((x1+pad, ty+pad), text, fill=(255, 255, 255), font=font)

        # Aggiunge bbox pixel al record per il report
        win["bbox"] = [x1, y1, x2, y2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def run_analyze():
    if "image" not in request.files:
        return jsonify({"error": "Nessuna immagine ricevuta"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Nome file vuoto"}), 400
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXT:
        return jsonify({"error": f"Formato non supportato: {suffix}"}), 400

    run_id = str(uuid.uuid4())[:8]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)

    orig_path = run_dir / f"original{suffix}"
    file.save(orig_path)

    out_img  = run_dir / "analyzed.jpg"
    out_json = run_dir / "report.json"

    conf_large = float(request.form.get("conf_large", CONF_LARGE))
    conf_small = float(request.form.get("conf_small", CONF_SMALL))
    large_size = int(request.form.get("large_size", LARGE_SIZE))
    min_size   = int(request.form.get("min_size", MIN_CROP_SIZE))

    try:
        analyze(orig_path, out_img, out_json, conf_large, min_size,
                conf_large, conf_small, large_size)
        report = json.loads(out_json.read_text("utf-8"))
        report["source"] = "local_model"
    except Exception as exc:
        return jsonify({"error": f"Errore pipeline: {exc}"}), 500

    # Fallback Azure GPT-4o se 0 finestre rilevate
    if report.get("total", 0) == 0:
        gpt = analyze_with_gpt4o(orig_path)
        report.update(gpt)
        # Disegna i box GPT sull'immagine originale
        if gpt.get("windows"):
            gpt_dims = (gpt.get("_gpt_w"), gpt.get("_gpt_h"))
            draw_gpt_boxes(orig_path, out_img, gpt["windows"], gpt_dims=gpt_dims)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")

    # Salva storico
    history = load_history()
    history.insert(0, {
        "id": run_id,
        "filename": file.filename,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total": report.get("total", 0),
        "clean": report.get("clean", 0),
        "dirty": report.get("dirty", 0),
        "source": report.get("source", "local_model"),
    })
    save_history(history)

    result_img = out_img if out_img.exists() else orig_path
    img_b64 = base64.b64encode(result_img.read_bytes()).decode()

    return jsonify({"id": run_id, "report": report, "image_b64": img_b64})


@app.route("/history")
def get_history():
    return jsonify(load_history())


@app.route("/result/<run_id>/report")
def get_report(run_id):
    path = RUNS_DIR / run_id / "report.json"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify(json.loads(path.read_text("utf-8")))


@app.route("/result/<run_id>/image")
def result_image(run_id):
    run_dir = RUNS_DIR / run_id
    img = run_dir / "analyzed.jpg"
    if not img.exists():
        originals = [f for f in run_dir.iterdir() if f.stem == "original"]
        if originals:
            img = originals[0]
    return send_file(img)


@app.route("/result/<run_id>/download/json")
def download_json(run_id):
    return send_file(
        RUNS_DIR / run_id / "report.json",
        as_attachment=True,
        download_name=f"report_{run_id}.json",
    )


@app.route("/result/<run_id>", methods=["DELETE"])
def delete_run(run_id):
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        return jsonify({"error": "not found"}), 404
    shutil.rmtree(run_dir)
    history = [h for h in load_history() if h["id"] != run_id]
    save_history(history)
    return jsonify({"ok": True})


@app.route("/history", methods=["DELETE"])
def delete_all():
    for run_dir in RUNS_DIR.iterdir():
        if run_dir.is_dir() and run_dir.name != "history.json":
            shutil.rmtree(run_dir, ignore_errors=True)
    save_history([])
    return jsonify({"ok": True})


@app.route("/result/<run_id>/download/image")
def download_image(run_id):
    run_dir = RUNS_DIR / run_id
    img = run_dir / "analyzed.jpg"
    if not img.exists():
        originals = [f for f in run_dir.iterdir() if f.stem == "original"]
        if originals:
            img = originals[0]
    return send_file(img, as_attachment=True, download_name=f"analyzed_{run_id}.jpg")


if __name__ == "__main__":
    print(f"\n{'='*52}")
    print(f"  AI CHALLENGE BICCHIERI TEAM")
    print(f"  http://localhost:5050")
    print(f"{'='*52}\n")
    app.run(debug=True, port=5050, host="0.0.0.0")
