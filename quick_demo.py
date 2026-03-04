"""
ScribbleNet — Modern Handwriting Recognition App
Powered by microsoft/trocr-base-handwritten

Run with:
    streamlit run quick_demo.py
"""

import csv
import io
import json
import time
from io import BytesIO
from pathlib import Path

import httpx
import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from streamlit_drawable_canvas import st_canvas
from transformers import AutoProcessor, VisionEncoderDecoderModel

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_ID  = "microsoft/trocr-base-handwritten"
MODEL_DIR = Path(__file__).parent / "models" / "trocr-base-handwritten"

st.set_page_config(
    page_title="ScribbleNet",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp { background: #0f0f13; color: #e8e8f0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #16161d !important;
    border-right: 1px solid #2a2a38;
}
[data-testid="stSidebar"] * { color: #c8c8d8 !important; }

/* Logo header */
.app-header {
    display: flex; align-items: center; gap: 14px;
    padding: 6px 0 24px 0;
}
.app-logo {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}
.app-sub { font-size: 0.78rem; color: #6b6b88; font-weight: 400; }

/* Upload zone */
.upload-zone {
    border: 2px dashed #2e2e44;
    border-radius: 16px;
    padding: 40px 20px;
    text-align: center;
    background: #13131a;
    transition: border-color .2s;
    margin-bottom: 16px;
}
.upload-zone:hover { border-color: #818cf8; }

/* Result card */
.result-card {
    background: #16161d;
    border: 1px solid #2a2a3a;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 10px 0;
}
.result-text {
    font-size: 2rem; font-weight: 600;
    color: #e8e8f8;
    letter-spacing: 0.5px;
    word-break: break-all;
    line-height: 1.3;
}
.result-label {
    font-size: 0.72rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 1.5px;
    color: #5a5a78; margin-bottom: 8px;
}

/* Confidence bar */
.conf-wrap { margin-top: 14px; }
.conf-bar-bg {
    background: #1e1e2e; border-radius: 100px;
    height: 6px; width: 100%; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    transition: width .6s ease;
}
.conf-label {
    display: flex; justify-content: space-between;
    font-size: 0.75rem; color: #5a5a78;
    margin-top: 5px;
}

/* History item */
.hist-item {
    background: #13131a;
    border: 1px solid #222230;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex; justify-content: space-between; align-items: center;
    cursor: pointer; transition: border-color .15s;
}
.hist-item:hover { border-color: #818cf8; }
.hist-word { font-size: 1rem; font-weight: 500; color: #d0d0e8; }
.hist-meta { font-size: 0.72rem; color: #4a4a68; }

/* Mode pill */
.mode-pill {
    display: inline-block;
    background: linear-gradient(135deg, #1e1e30, #252538);
    border: 1px solid #3a3a54;
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.75rem; color: #818cf8;
    font-weight: 500; letter-spacing: 0.5px;
    margin-bottom: 12px;
}

/* Action button */
.stButton > button {
    background: linear-gradient(135deg, #818cf8, #c084fc) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
    padding: 10px 24px !important;
    transition: opacity .2s, transform .1s !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.88 !important; transform: translateY(-1px) !important;
}

/* Secondary buttons */
.stDownloadButton > button {
    background: #1e1e2e !important;
    color: #818cf8 !important;
    border: 1px solid #3a3a54 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
}

/* Divider */
hr { border-color: #1e1e2e !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #13131a;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #222230;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    color: #5a5a78 !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    padding: 6px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #818cf850, #c084fc30) !important;
    color: #c084fc !important;
}

/* Metric */
[data-testid="metric-container"] {
    background: #16161d; border: 1px solid #2a2a3a;
    border-radius: 12px; padding: 14px 18px;
}
[data-testid="metric-container"] label { color: #5a5a78 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #c084fc !important; font-size: 1.5rem !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #818cf8 !important; }

/* Text area */
textarea {
    background: #13131a !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 10px !important;
    color: #d0d0e8 !important;
    font-family: 'Inter', monospace !important;
}

/* Selectbox / slider */
[data-baseweb="select"] { background: #13131a !important; }
[data-testid="stSlider"] { color: #818cf8 !important; }

/* Badge */
.badge {
    display: inline-block;
    background: #1e1e30; border: 1px solid #3a3a54;
    border-radius: 6px; padding: 2px 10px;
    font-size: 0.72rem; color: #818cf8; font-weight: 500;
}
.toast {
    background: #1e2e1e; border: 1px solid #2a4a2a;
    border-radius: 10px; padding: 10px 16px;
    color: #6fcf97; font-size: 0.85rem;
    margin-top: 8px;
}

/* Form digitizer table */
.form-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
.form-table th {
    text-align: left; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1.2px;
    color: #5a5a78; padding: 8px 14px;
    border-bottom: 1px solid #2a2a3a;
    background: #13131a;
}
.form-table td {
    padding: 10px 14px; font-size: 0.9rem;
    border-bottom: 1px solid #1e1e2e;
    vertical-align: top;
}
.form-table td:first-child { color: #818cf8; font-weight: 500; width: 38%; }
.form-table td:last-child  { color: #d0d0e8; }
.form-table tr:last-child td { border-bottom: none; }
.form-table tr:hover td { background: #16161d; }

/* API key input override */
input[type="password"] {
    background: #13131a !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 8px !important;
    color: #d0d0e8 !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    # Always load from local disk — avoids the huggingface_hub httpx closed-client
    # bug that occurs inside Streamlit sessions. Run download_model.py once first.
    if not MODEL_DIR.exists():
        st.error(
            "Model not found locally. Run this once in your terminal first:\n\n"
            "```\npython download_model.py\n```"
        )
        st.stop()
    src = str(MODEL_DIR)
    processor = AutoProcessor.from_pretrained(src, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(src, local_files_only=True)
    model.eval()
    return processor, model


def run_ocr(image: Image.Image, processor, model, beam_width: int = 4) -> tuple[str, float]:
    img = image.convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    with torch.no_grad():
        output = model.generate(
            pixel_values,
            max_length=32,
            num_beams=beam_width,
            output_scores=True,
            return_dict_in_generate=True,
        )
    text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
    if output.scores:
        probs = [torch.softmax(s, dim=-1).max().item() for s in output.scores]
        confidence = sum(probs) / len(probs)
    else:
        confidence = 0.0
    return text, confidence


def preprocess_image(image: Image.Image, enhance_contrast: bool, denoise: bool) -> Image.Image:
    img = image.convert("L").convert("RGB")
    if enhance_contrast:
        img = ImageEnhance.Contrast(img).enhance(2.0)
    if denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))
    return img


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


# ── Mistral / Form helpers ───────────────────────────────────────────────────

def call_mistral(ocr_text: str, api_key: str) -> list[dict]:
    """Send OCR text to Mistral and return [{field, value}, ...]"""
    system = (
        "You are a form data extraction assistant. "
        "Given raw OCR text from a handwritten form, identify every field label and its filled-in value. "
        "Return ONLY a JSON array where each element has exactly two string keys: "
        '"field" (snake_case label, e.g. first_name) and "value" (the written value). '
        "If a value is blank, use an empty string. No explanation, no markdown fences."
    )
    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Extract all form fields from this OCR text:\n\n{ocr_text}"},
        ],
        "temperature": 0.1,
    }
    resp = httpx.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    # Strip optional markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def fields_to_html_page(fields: list[dict], title: str = "Form Data") -> str:
    rows = "".join(
        f'<tr><td>{r["field"]}</td><td>{r["value"]}</td></tr>'
        for r in fields
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{title}</title>
<style>
  body{{font-family:system-ui,sans-serif;background:#f9f9f9;padding:40px;color:#1a1a1a}}
  h1{{font-size:1.4rem;margin-bottom:24px;color:#333}}
  table{{border-collapse:collapse;width:100%;max-width:640px;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 6px #0001}}
  th{{text-align:left;padding:10px 16px;font-size:.75rem;text-transform:uppercase;letter-spacing:1px;color:#666;border-bottom:1px solid #eee;background:#fafafa}}
  td{{padding:11px 16px;font-size:.95rem;border-bottom:1px solid #f0f0f0}}
  td:first-child{{color:#5b50e8;font-weight:600;width:38%}}
  tr:last-child td{{border-bottom:none}}
</style>
</head><body>
<h1>{title}</h1>
<table><thead><tr><th>Field</th><th>Value</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>"""


def fields_to_csv(fields: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([r["field"] for r in fields])
    w.writerow([r["value"] for r in fields])
    return buf.getvalue()


def fields_to_sql(fields: list[dict], table: str = "form_data") -> str:
    cols   = ", ".join(r["field"] for r in fields)
    vals   = ", ".join(
        "'" + r["value"].replace("'", "''") + "'"
        for r in fields
    )
    return f"INSERT INTO {table} ({cols})\nVALUES ({vals});"


def confidence_html(conf: float) -> str:
    pct = conf * 100
    color = "#6fcf97" if pct >= 70 else "#f2c94c" if pct >= 40 else "#eb5757"
    return f"""
    <div class="conf-wrap">
        <div class="result-label">Confidence</div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct:.1f}%; background: {color};"></div>
        </div>
        <div class="conf-label"><span>{pct:.1f}%</span><span>{"High" if pct>=70 else "Medium" if pct>=40 else "Low"}</span></div>
    </div>
    """


# ── Session State ─────────────────────────────────────────────────────────────

if "history"      not in st.session_state: st.session_state.history      = []
if "last_result"  not in st.session_state: st.session_state.last_result  = None
if "image"        not in st.session_state: st.session_state.image        = None
if "form_fields"  not in st.session_state: st.session_state.form_fields  = None
if "form_ocr_raw" not in st.session_state: st.session_state.form_ocr_raw = ""


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="app-header">
        <div>
            <div class="app-logo">ScribbleNet</div>
            <div class="app-sub">Handwriting Recognition · v1.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="result-label">Recognition Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        label="mode",
        options=["🖼️ Full Image OCR", "✂️ Region Selection", "⚡ Batch OCR", "📋 Form Digitizer"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="result-label">Settings</div>', unsafe_allow_html=True)
    beam_width = st.slider("Beam Width", 1, 8, 4, help="Higher = more accurate but slower")
    enhance_contrast = st.toggle("Enhance Contrast", value=True)
    denoise = st.toggle("Denoise Image", value=False)
    mistral_key = st.text_input(
        "Mistral API Key",
        type="password",
        placeholder="sk-…  (for Form Digitizer)",
        help="Required only for Form Digitizer mode",
    )
    st.markdown("---")
    st.markdown('<div class="result-label">OCR History</div>', unsafe_allow_html=True)

    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history[-8:])):
            conf_pct = entry["confidence"] * 100
            color = "#6fcf97" if conf_pct >= 70 else "#f2c94c" if conf_pct >= 40 else "#eb5757"
            st.markdown(f"""
            <div class="hist-item">
                <span class="hist-word">{entry["text"] or "—"}</span>
                <span class="hist-meta" style="color:{color}">{conf_pct:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown('<div class="hist-meta" style="color:#3a3a58; font-size:0.8rem;">No recognitions yet.</div>', unsafe_allow_html=True)

    # Model status
    st.markdown("---")
    st.markdown('<div class="result-label">Model</div>', unsafe_allow_html=True)
    st.markdown('<span class="badge">trocr-base-handwritten</span>', unsafe_allow_html=True)


# ── Main Content ──────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
    <div>
        <span style="font-size:1.6rem; font-weight:700; color:#e8e8f8;">Handwriting</span>
        <span style="font-size:1.6rem; font-weight:300; color:#5a5a78;"> Recognition</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Mode pill
mode_clean = mode.split(" ", 1)[1]
st.markdown(f'<div class="mode-pill">{mode_clean}</div>', unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────

if mode not in ("⚡ Batch OCR",):
    uploaded = st.file_uploader(
        "Drop your image here",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        label_visibility="collapsed",
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        # Sensible max display size
        MAX_W = 1200
        if img.width > MAX_W:
            ratio = MAX_W / img.width
            img = img.resize((MAX_W, int(img.height * ratio)), Image.LANCZOS)
        st.session_state.image = img
else:
    uploaded_many = st.file_uploader(
        "Upload multiple images",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — Full Image OCR
# ─────────────────────────────────────────────────────────────────────────────

if mode == "🖼️ Full Image OCR":
    if st.session_state.image:
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown('<div class="result-label">Image Preview</div>', unsafe_allow_html=True)
            display_img = preprocess_image(st.session_state.image, enhance_contrast, denoise) if (enhance_contrast or denoise) else st.session_state.image
            st.image(display_img, use_container_width=True)

            col_run, col_dl = st.columns(2)
            with col_run:
                run = st.button("Recognize Text", use_container_width=True)
            with col_dl:
                st.download_button(
                    "Save Image",
                    data=image_to_bytes(display_img),
                    file_name="processed.png",
                    mime="image/png",
                    use_container_width=True,
                )

        with right:
            st.markdown('<div class="result-label">Image Info</div>', unsafe_allow_html=True)
            img = st.session_state.image
            c1, c2 = st.columns(2)
            c1.metric("Width", f"{img.width}px")
            c2.metric("Height", f"{img.height}px")

            st.markdown("<br>", unsafe_allow_html=True)

            if run or st.session_state.last_result:
                if run:
                    with st.spinner(""):
                        processor, model = load_model()
                        proc_img = preprocess_image(img, enhance_contrast, denoise)
                        text, conf = run_ocr(proc_img, processor, model, beam_width)
                        st.session_state.last_result = {"text": text, "confidence": conf}
                        st.session_state.history.append({"text": text, "confidence": conf})

                result = st.session_state.last_result

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<div class="result-label">Extracted Text</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-text">{result["text"] or "Nothing detected"}</div>', unsafe_allow_html=True)
                st.markdown(confidence_html(result["confidence"]), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                edited = st.text_area("Edit result", value=result["text"], height=80, label_visibility="collapsed", placeholder="Edit recognized text here...")

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button("⬇ Download TXT", data=edited.encode(), file_name="result.txt", mime="text/plain", use_container_width=True)
                with dl2:
                    st.download_button("⬇ Download JSON", data=json.dumps({"text": edited, "confidence": result["confidence"]}, indent=2).encode(), file_name="result.json", mime="application/json", use_container_width=True)
            else:
                st.markdown("""
                <div style="padding:40px 20px; text-align:center; color:#3a3a58;">
                    <div style="font-size:2.5rem; margin-bottom:12px;">🔍</div>
                    <div style="font-size:0.9rem;">Click <b>Recognize Text</b> to extract handwriting</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:2.5rem; margin-bottom:12px;">✍️</div>
            <div style="font-size:1rem; color:#4a4a68; margin-bottom:6px;">Drop a handwritten image here</div>
            <div style="font-size:0.8rem; color:#3a3a54;">PNG · JPG · TIFF · BMP</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — Region Selection
# ─────────────────────────────────────────────────────────────────────────────

elif mode == "✂️ Region Selection":
    if st.session_state.image:
        img = st.session_state.image

        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown('<div class="result-label" style="margin-bottom:6px;">Draw a rectangle around the text region</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.78rem; color:#4a4a68; margin-bottom:10px;">Click and drag on the image to select any area to recognize.</div>', unsafe_allow_html=True)

            # Scale for display
            CANVAS_W = 700
            scale = CANVAS_W / img.width
            canvas_h = int(img.height * scale)
            display_img = img.resize((CANVAS_W, canvas_h), Image.LANCZOS)

            canvas_result = st_canvas(
                fill_color="rgba(129, 140, 248, 0.08)",
                stroke_color="#818cf8",
                stroke_width=2,
                background_image=display_img,
                update_streamlit=True,
                height=canvas_h,
                width=CANVAS_W,
                drawing_mode="rect",
                key="canvas",
            )

        with right:
            st.markdown('<div class="result-label">Region OCR</div>', unsafe_allow_html=True)

            has_rect = (
                canvas_result.json_data is not None
                and len(canvas_result.json_data.get("objects", [])) > 0
            )

            if has_rect:
                obj = canvas_result.json_data["objects"][-1]
                x = int(obj["left"] / scale)
                y = int(obj["top"] / scale)
                w = int(obj["width"] / scale)
                h = int(obj["height"] / scale)

                # Clamp to bounds
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img.width, x + w)
                y2 = min(img.height, y + h)

                if x2 > x1 and y2 > y1:
                    crop = img.crop((x1, y1, x2, y2))
                    crop_proc = preprocess_image(crop, enhance_contrast, denoise)

                    st.image(crop_proc, caption="Selected Region", use_container_width=True)
                    st.markdown(f'<div style="font-size:0.75rem; color:#4a4a68; margin-top:4px;">Region: ({x1}, {y1}) → ({x2}, {y2}) · {x2-x1}×{y2-y1}px</div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Recognize Region", use_container_width=True):
                        with st.spinner(""):
                            processor, model_obj = load_model()
                            text, conf = run_ocr(crop_proc, processor, model_obj, beam_width)

                        st.session_state.history.append({"text": text, "confidence": conf})

                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown('<div class="result-label">Extracted Text</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="result-text">{text or "Nothing detected"}</div>', unsafe_allow_html=True)
                        st.markdown(confidence_html(conf), unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        edited = st.text_area("Edit", value=text, height=80, label_visibility="collapsed")
                        st.download_button("⬇ Download TXT", data=edited.encode(), file_name="region_result.txt", mime="text/plain", use_container_width=True)
                else:
                    st.warning("Selection too small — draw a larger region.")
            else:
                st.markdown("""
                <div style="padding:50px 20px; text-align:center; color:#3a3a58;">
                    <div style="font-size:2rem; margin-bottom:10px;">✂️</div>
                    <div style="font-size:0.88rem;">Draw a rectangle on the image<br>to select a region</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:2.5rem; margin-bottom:12px;">✂️</div>
            <div style="font-size:1rem; color:#4a4a68; margin-bottom:6px;">Upload an image first</div>
            <div style="font-size:0.8rem; color:#3a3a54;">Then draw a box around any text region</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — Batch OCR
# ─────────────────────────────────────────────────────────────────────────────

elif mode == "⚡ Batch OCR":
    if uploaded_many:
        st.markdown(f'<div style="font-size:0.85rem; color:#5a5a78; margin-bottom:16px;">{len(uploaded_many)} image(s) loaded</div>', unsafe_allow_html=True)

        if st.button(f"Recognize All {len(uploaded_many)} Images", use_container_width=False):
            processor, model_obj = load_model()
            results = []

            progress = st.progress(0, text="Starting...")
            cols_batch = st.columns(min(len(uploaded_many), 4))

            for i, f in enumerate(uploaded_many):
                img_b = Image.open(f).convert("RGB")
                proc_b = preprocess_image(img_b, enhance_contrast, denoise)
                text, conf = run_ocr(proc_b, processor, model_obj, beam_width)
                results.append({"file": f.name, "text": text, "confidence": round(conf, 4)})
                st.session_state.history.append({"text": text, "confidence": conf})

                with cols_batch[i % len(cols_batch)]:
                    st.image(img_b, use_container_width=True)
                    conf_pct = conf * 100
                    c = "#6fcf97" if conf_pct >= 70 else "#f2c94c" if conf_pct >= 40 else "#eb5757"
                    st.markdown(f'<div style="font-size:0.95rem; font-weight:600; color:#d0d0e8;">{text or "—"}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:0.75rem; color:{c};">{conf_pct:.0f}% · {f.name}</div>', unsafe_allow_html=True)

                progress.progress((i + 1) / len(uploaded_many), text=f"Processing {f.name}...")

            progress.empty()
            st.markdown('<div class="toast">✓ Batch complete</div>', unsafe_allow_html=True)

            st.markdown("---")
            batch_json = json.dumps(results, indent=2)
            batch_txt = "\n".join([f'{r["file"]}: {r["text"]}' for r in results])

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇ Download All (TXT)", data=batch_txt.encode(), file_name="batch_results.txt", mime="text/plain", use_container_width=True)
            with c2:
                st.download_button("⬇ Download All (JSON)", data=batch_json.encode(), file_name="batch_results.json", mime="application/json", use_container_width=True)
    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:2.5rem; margin-bottom:12px;">⚡</div>
            <div style="font-size:1rem; color:#4a4a68; margin-bottom:6px;">Upload multiple images at once</div>
            <div style="font-size:0.8rem; color:#3a3a54;">All images will be recognized in one batch</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 4 — Form Digitizer
# ─────────────────────────────────────────────────────────────────────────────

elif mode == "📋 Form Digitizer":
    if st.session_state.image:
        img = st.session_state.image
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown('<div class="result-label">Form Preview</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)

            run_form = st.button("✦ OCR + Parse Form", use_container_width=True)

        with right:
            st.markdown('<div class="result-label">Extracted Fields</div>', unsafe_allow_html=True)

            if run_form:
                if not mistral_key:
                    st.error("Enter your Mistral API key in the sidebar Settings first.")
                else:
                    with st.spinner("Running OCR…"):
                        processor, model_obj = load_model()
                        proc_img = preprocess_image(img, enhance_contrast, denoise)
                        ocr_text, conf = run_ocr(proc_img, processor, model_obj, beam_width)
                        st.session_state.form_ocr_raw = ocr_text
                        st.session_state.history.append({"text": ocr_text, "confidence": conf})

                    with st.spinner("Parsing with Mistral…"):
                        try:
                            fields = call_mistral(ocr_text, mistral_key)
                            st.session_state.form_fields = fields
                        except Exception as exc:
                            st.error(f"Mistral error: {exc}")
                            st.session_state.form_fields = None

            # ── Raw OCR preview ───────────────────────────────────────────
            if st.session_state.form_ocr_raw:
                with st.expander("Raw OCR text", expanded=False):
                    st.code(st.session_state.form_ocr_raw, language=None)

            # ── Structured results ────────────────────────────────────────
            fields = st.session_state.form_fields
            if fields:
                table_rows = "".join(
                    f'<tr><td>{r["field"]}</td><td>{r["value"]}</td></tr>'
                    for r in fields
                )
                st.markdown(f"""
                <div class="result-card" style="padding:0; overflow:hidden;">
                    <table class="form-table">
                    <thead><tr><th>Field</th><th>Value</th></tr></thead>
                    <tbody>{table_rows}</tbody>
                    </table>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Editable JSON for tweaking ────────────────────────────
                edited_json = st.text_area(
                    "Edit fields (JSON)",
                    value=json.dumps(fields, indent=2),
                    height=180,
                    label_visibility="collapsed",
                )
                try:
                    fields_final = json.loads(edited_json)
                except json.JSONDecodeError:
                    fields_final = fields
                    st.warning("Invalid JSON — using last valid version.")

                st.markdown("---")

                # ── Downloads ─────────────────────────────────────────────
                html_out = fields_to_html_page(fields_final)
                csv_out  = fields_to_csv(fields_final)
                sql_out  = fields_to_sql(fields_final)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.download_button(
                        "⬇ HTML",
                        data=html_out.encode(),
                        file_name="form_output.html",
                        mime="text/html",
                        use_container_width=True,
                    )
                with c2:
                    st.download_button(
                        "⬇ CSV",
                        data=csv_out.encode(),
                        file_name="form_output.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with c3:
                    st.download_button(
                        "⬇ SQL",
                        data=sql_out.encode(),
                        file_name="form_output.sql",
                        mime="text/plain",
                        use_container_width=True,
                    )

                # ── Inline SQL preview ────────────────────────────────────
                st.markdown(
                    f'<div style="margin-top:8px; font-size:0.72rem; color:#4a4a68;">SQL preview</div>',
                    unsafe_allow_html=True,
                )
                st.code(sql_out, language="sql")

            elif not run_form:
                st.markdown("""
                <div style="padding:50px 20px; text-align:center; color:#3a3a58;">
                    <div style="font-size:2.5rem; margin-bottom:12px;">📋</div>
                    <div style="font-size:0.9rem;">Click <b>OCR + Parse Form</b><br>to extract and structure the fields</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:2.5rem; margin-bottom:12px;">📋</div>
            <div style="font-size:1rem; color:#4a4a68; margin-bottom:6px;">Upload a handwritten form</div>
            <div style="font-size:0.8rem; color:#3a3a54;">ScribbleNet OCRs it · Mistral structures it · You export it</div>
        </div>
        """, unsafe_allow_html=True)

