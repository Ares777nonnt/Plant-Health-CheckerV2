import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ============================
# Optional Torch
# ============================
TORCH_AVAILABLE = False
try:
    import torch
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ============================
# Page config
# ============================
st.set_page_config(
    page_title="Plant Health Checker – Hybrid AI",
    page_icon="🌿",
    layout="wide"
)

# ============================
# CSS (palette preferita)
# ============================
st.markdown("""
<style>
.stApp {
  background: linear-gradient(135deg, #001a17 0%, #0a3d35 100%);
  color: #eafffb;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
.block-container {
  max-width: 1200px;
  padding-top: 2.2rem;
}
.phc-hero {
  border-radius: 18px;
  padding: 22px;
  background: rgba(0,0,0,0.25);
  border: 1px solid rgba(140,255,230,0.18);
  margin-bottom: 1.8rem;
}
.phc-card {
  background: rgba(0,0,0,0.22);
  border: 1px solid rgba(140,255,230,0.18);
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 1.4rem;
}
.small-muted {
  color: #bfeee6;
  font-size: 0.9rem;
}
.stButton > button {
  border-radius: 12px;
  border: none;
  font-weight: 650;
  padding: 0.6rem 1.2rem;
  background: linear-gradient(90deg, #2ef2c8, #00d1ff);
  color: #001a17;
}
.stButton > button:disabled {
  background: rgba(200,200,200,0.2);
  color: #88aaa5;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(90deg, #2ef2c8, #00d1ff);
  color: #001a17 !important;
  border-radius: 10px;
}
.stProgress > div > div {
  background-color: #2ef2c8;
}
[data-testid="stDataFrame"] {
  background: rgba(0,0,0,0.2);
  border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ============================
# Session state
# ============================
if "records" not in st.session_state:
    st.session_state.records = []

if "last_image" not in st.session_state:
    st.session_state.last_image = None

if "image_protocol_ok" not in st.session_state:
    st.session_state.image_protocol_ok = False

if "last_eval" not in st.session_state:
    st.session_state.last_eval = None

if "guided_mode" not in st.session_state:
    st.session_state.guided_mode = True

if "guided_step" not in st.session_state:
    st.session_state.guided_step = 1


# ============================
# Utilities
# ============================
def generate_sample_id():
    year = datetime.now().year
    n = len(st.session_state.records) + 1
    return f"PHC_{year}_{n:06d}"


def export_excel(rows):
    df = pd.DataFrame(rows)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="HybridAI")
    out.seek(0)
    return out.read()


# ============================
# Physiology
# ============================
def physio_rule_score(p):
    score = 0
    reasons = []

    if p["fvfm"] >= 0.8: score += 2
    else: reasons.append("Low Fv/Fm (PSII efficiency)")

    if p["chltot"] >= 1.5: score += 2
    else: reasons.append("Low chlorophyll content")

    if p["cartot"] >= 0.5: score += 2
    else: reasons.append("Low carotenoid content")

    if p["spad"] >= 40: score += 2
    else: reasons.append("Low SPAD")

    if p["qp"] >= 0.7: score += 2
    else: reasons.append("Low photochemical quenching")

    if 0.3 <= p["qn"] <= 0.7: score += 2
    else: reasons.append("Altered non-photochemical quenching")

    if score >= 10: klass = "Healthy"
    elif score >= 6: klass = "Moderate stress"
    else: klass = "High stress"

    return score, klass, reasons


def physio_prior(score):
    s = float(score)
    p_h = 1 / (1 + np.exp(-(s - 9)))
    p_hi = 1 / (1 + np.exp((s - 5)))
    p_m = max(0, 1 - (p_h + p_hi))
    vec = np.array([p_h, p_m, p_hi])
    return vec / vec.sum()


# ============================
# Image analysis
# ============================
@st.cache_resource(show_spinner=False)
def load_cnn():
    w = EfficientNet_B0_Weights.DEFAULT
    m = efficientnet_b0(weights=w)
    m.eval()
    m.classifier = torch.nn.Identity()
    return m, w.transforms()


def image_risk(img):
    if TORCH_AVAILABLE:
        m, tf = load_cnn()
        x = tf(img).unsqueeze(0)
        with torch.no_grad():
            f = m(x).squeeze().numpy()
        risk = float(np.clip(1 - f.std() * 3, 0, 1))
        method = "CNN features"
    else:
        arr = np.array(img).astype(float)
        g = arr[...,1].mean()/255
        b = arr.mean()/255
        risk = float(np.clip(1 - (g + b)/2, 0, 1))
        method = "Color heuristic"
    return risk, method


def fuse_probs(p_phys, p_img, w=0.7):
    lp = w*np.log(p_phys+1e-8)+(1-w)*np.log(p_img+1e-8)
    p = np.exp(lp-lp.max())
    return p/p.sum()


# ============================
# HERO + Guided toggle
# ============================
st.markdown("""
<div class="phc-hero">
  <h2>Plant Health Checker – Hybrid AI</h2>
  <div class="small-muted">
    Step-by-step hybrid assessment using leaf image + physiological parameters
  </div>
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns([0.7,0.3])
with col_a:
    st.caption("Guided Mode helps first-time users follow the correct workflow.")
with col_b:
    st.session_state.guided_mode = st.toggle(
        "Guided mode",
        value=st.session_state.guided_mode
    )


# ============================
# Tabs
# ============================
tabs = st.tabs([
    "1. Upload image",
    "2. Evaluate & preview",
    "3. Saved records"
])

has_image = st.session_state.last_image is not None
has_preview = st.session_state.last_eval is not None
protocol_ok = st.session_state.image_protocol_ok


# ============================
# TAB 1 — IMAGE
# ============================
with tabs[0]:
    st.markdown('<div class="phc-card">', unsafe_allow_html=True)

    st.info("Step 1 — Upload a single leaf image to enable hybrid analysis.")

    st.markdown("""
    **Recommended conditions**
    - Single leaf
    - Neutral background
    - Diffuse natural light
    - No flash / no shadows
    """)

    st.session_state.image_protocol_ok = st.checkbox(
        "I confirm the image follows the recommended protocol"
    )

    up = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])
    if up:
        img = Image.open(up).convert("RGB")
        st.session_state.last_image = img
        st.image(img, use_container_width=True)
        if st.session_state.guided_mode:
            st.session_state.guided_step = 2
            st.success("Image uploaded. Proceed to Step 2.")

    st.markdown('</div>', unsafe_allow_html=True)


# ============================
# TAB 2 — EVALUATE
# ============================
with tabs[1]:
    st.markdown('<div class="phc-card">', unsafe_allow_html=True)

    if st.session_state.guided_mode and not has_image:
        st.warning("Upload an image in Step 1 to enable evaluation.")

    st.info("Step 2 — Insert physiological parameters and preview the result.")

    fvfm = st.number_input("Fv/Fm",0.0,1.0,0.8,0.01)
    chltot = st.number_input("Chl TOT",0.0,5.0,1.5,0.01)
    cartot = st.number_input("CAR TOT",0.0,5.0,0.5,0.01)
    spad = st.number_input("SPAD",0.0,100.0,40.0,0.5)
    qp = st.number_input("qp",0.0,1.0,0.7,0.01)
    qn = st.number_input("qN",0.0,1.0,0.5,0.01)

    can_eval = has_image or not st.session_state.guided_mode

    if st.button("Evaluate (preview only)", disabled=not can_eval):
        p = dict(fvfm=fvfm,chltot=chltot,cartot=cartot,spad=spad,qp=qp,qn=qn)
        score, cls, reasons = physio_rule_score(p)
        p_phys = physio_prior(score)

        risk, method = image_risk(st.session_state.last_image)
        p_img = np.array([1-risk, 0.5*(1-risk), risk])
        p_img = p_img/p_img.sum()

        fused = fuse_probs(p_phys, p_img)
        classes = ["Healthy","Moderate stress","High stress"]
        idx = int(np.argmax(fused))

        st.session_state.last_eval = {
            "Prediction": classes[idx],
            "Confidence": float(fused[idx]),
            "PhysioScore": score,
            "VisualRisk": risk,
            "ImageMethod": method
        }

        if st.session_state.guided_mode:
            st.session_state.guided_step = 3

    if has_preview:
        ev = st.session_state.last_eval
        st.markdown("### Preview result")
        st.write(ev)
        st.progress(int(ev["Confidence"]*100))

        can_save = has_image and protocol_ok
        if st.button("Save record", disabled=not can_save):
            row = {
                "Sample_ID": generate_sample_id(),
                **ev
            }
            st.session_state.records.append(row)
            st.success("Record saved.")

    st.caption("This is a research tool, not a diagnostic system.")
    st.markdown('</div>', unsafe_allow_html=True)


# ============================
# TAB 3 — RECORDS
# ============================
with tabs[2]:
    st.markdown('<div class="phc-card">', unsafe_allow_html=True)
    st.info("Step 3 — Review and export saved records.")

    if st.session_state.records:
        df = pd.DataFrame(st.session_state.records)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download Excel",
            export_excel(st.session_state.records),
            "plant_health_records.xlsx"
        )
    else:
        st.info("No records saved yet.")

    st.markdown('</div>', unsafe_allow_html=True)
