import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os

st.set_page_config(page_title="Digit Recognizer", page_icon="🔢", layout="wide")

# ── CSS — injected AFTER set_page_config, kept simple & safe ─────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [data-testid="stApp"] {
    background-color: #0d1117 !important;
    font-family: 'Inter', sans-serif;
    color: #e6edf3;
}

/* gradient blobs */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 70% 50% at 15% 10%, rgba(88,101,242,0.22) 0%, transparent 55%),
        radial-gradient(ellipse 55% 45% at 85% 85%, rgba(16,185,129,0.16) 0%, transparent 50%),
        #0d1117 !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1080px !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header { display: none !important; }

/* ── headings ── */
h1 { font-size: 2.6rem !important; font-weight: 800 !important; letter-spacing: -0.03em !important;
     background: linear-gradient(120deg, #fff 20%, #a5b4fc 60%, #6ee7b7);
     -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
     background-clip: text !important; }
h2 { font-size: 1rem !important; font-weight: 600 !important; color: #94a3b8 !important;
     letter-spacing: 0.08em !important; text-transform: uppercase !important; }

/* ── cards via st.container — target stVerticalBlock children ── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(22,33,55,0.85) !important;
    border: 1px solid rgba(99,102,241,0.22) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(8px) !important;
}

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.07) !important;
    border: 2px dashed rgba(99,102,241,0.35) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.7) !important;
    background: rgba(99,102,241,0.12) !important;
}
[data-testid="stFileUploaderDropzone"] { padding: 1.5rem !important; }

/* ── metric boxes ── */
[data-testid="stMetric"] {
    background: rgba(99,102,241,0.1) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #6ee7b7, #6366f1);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
}

/* ── progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6366f1, #10b981) !important;
    border-radius: 99px !important;
}
[data-testid="stProgressBar"] > div {
    background: rgba(99,102,241,0.12) !important;
    border-radius: 99px !important;
    height: 10px !important;
}

/* ── image ── */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
}

/* ── info / success / warning ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
}

/* ── caption / small text ── */
[data-testid="stCaptionContainer"] p { color: #475569 !important; font-size: 0.8rem !important; }

/* ── divider ── */
hr { border-color: rgba(99,102,241,0.15) !important; margin: 1.5rem 0 !important; }

/* ── probability rows ── */
.prob-row {
    display: flex; align-items: center; gap: 10px;
    padding: 4px 0;
}
.prob-digit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem; font-weight: 700;
    width: 22px; text-align: right; flex-shrink: 0;
}
.prob-track {
    flex: 1; height: 8px;
    background: rgba(99,102,241,0.1);
    border-radius: 99px; overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 99px; }
.prob-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    width: 44px; text-align: right; flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in [here, os.getcwd()]:
        for name in ["mnist_cnn_model.keras", "mnist_cnn_model.h5"]:
            p = os.path.join(folder, name)
            if os.path.exists(p):
                return tf.keras.models.load_model(p)
    return None

def preprocess(img):
    img = img.convert("L")
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 28, 28, 1)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.title("Handwritten Digit Recognizer")
st.caption("CNN trained on MNIST · Upload a digit image and the model reads it instantly")
st.markdown("---")

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    model = load_model()

if model is None:
    st.error("Model not found. Make sure `mnist_cnn_model.h5` is in the same folder as `app.py`.")
    st.stop()

# ── Stat bar ──────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Architecture", "CNN")
c2.metric("Training Images", "60,000")
c3.metric("Test Accuracy", "99%+")

st.markdown("---")

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    with st.container(border=True):
        st.subheader("Upload Image")
        uploaded = st.file_uploader(
            "Drop an image or click to browse",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            label_visibility="collapsed"
        )
        st.markdown(
            "<p style='font-size:0.82rem;color:#64748b;margin-top:0.6rem;'>"
            "💡 Works best with a <b style='color:#6ee7b7'>dark background</b> and white digit. "
            "Light backgrounds are auto-inverted.</p>",
            unsafe_allow_html=True
        )

    if uploaded:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("Preview")
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)
            st.caption(f"{img.width} × {img.height} px · {img.mode}")

with right:
    with st.container(border=True):
        st.subheader("Prediction")

        if not uploaded:
            st.markdown("""
            <div style='text-align:center;padding:3.5rem 0;'>
                <div style='font-family:JetBrains Mono,monospace;font-size:6rem;font-weight:700;
                            color:rgba(99,102,241,0.2);line-height:1;'>?</div>
                <p style='color:#334155;font-size:0.85rem;margin-top:0.8rem;'>
                    Upload an image to get started</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            img   = Image.open(uploaded)
            arr   = preprocess(img)
            probs = model.predict(arr, verbose=0)[0]
            pred  = int(np.argmax(probs))
            conf  = float(probs[pred]) * 100

            # big digit + confidence
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Predicted Digit", str(pred))
            m2.metric("Confidence", f"{conf:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(conf / 100)

            if conf < 60:
                st.warning(f"⚠️ Low confidence ({conf:.1f}%). Try a cleaner image.")

            # top-5 bar chart
            st.markdown("---")
            st.subheader("Top 5 Predictions")
            top5   = np.argsort(probs)[::-1][:5]
            max_p  = float(probs[top5[0]])
            colors = ["#6366f1", "#818cf8", "#a5b4fc", "#34d399", "#10b981"]

            rows = ""
            for rank, idx in enumerate(top5):
                p   = float(probs[idx])
                w   = (p / max_p) * 100
                c   = colors[rank]
                bold = "font-weight:800;" if rank == 0 else "font-weight:400;"
                rows += f"""
                <div class="prob-row">
                    <span class="prob-digit" style="color:{c};{bold}">{idx}</span>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{w:.1f}%;background:{c};"></div>
                    </div>
                    <span class="prob-pct" style="color:{c};{bold}">{p*100:.1f}%</span>
                </div>"""
            st.markdown(rows, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:0.75rem;color:#334155;"
    "font-family:JetBrains Mono,monospace;'>"
    "CNN · MNIST · TensorFlow · Streamlit</p>",
    unsafe_allow_html=True
)