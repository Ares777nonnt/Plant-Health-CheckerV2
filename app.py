
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ----------------------------
# Optional Torch
# ----------------------------
TORCH_AVAILABLE = False
try:
    import torch
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ----------------------------
# Page config + CSS
# ----------------------------
st.set_page_config(page_title="Plant Health Checker – Hybrid AI", page_icon="🌿", layout="wide")

CUSTOM_CSS = """
<style>

/* ===== BASE ===== */
.stApp {
  background-color: #0b1412;
  color: #e6f4f1;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}

.block-container {
  max-width: 1100px;
  padding-top: 2.2rem;
}

/* ===== TYPO ===== */
h1, h2, h3, h4 {
  color: #e6f4f1;
  font-weight: 600;
}

.small-muted {
  color: #9fbfba;
  font-size: 0.9rem;
}

/* ===== HERO ===== */
.phc-hero {
  background: none;
  border-bottom: 1px solid #1f2f2b;
  padding-bottom: 1.4rem;
  margin-bottom: 2rem;
}

/* ===== CARDS ===== */
.phc-card {
  background: #111d1a;
  border: 1px solid #1f2f2b;
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 1.4rem;
}

/* ===== INPUTS ===== */
.stTextInput input,
.stNumberInput input {
  background: #0b1412 !important;
  border: 1px solid #1f2f2b !important;
  border-radius: 8px !important;
  color: #e6f4f1 !important;
}

.stTextInput label,
.stNumberInput label {
  color: #cfe9e4;
  font-weight: 500;
}

/* ===== BUTTONS ===== */
.stButton > button {
  background: #2ef2c8;
  color: #0b1412;
  border: none;
  border-radius: 8px;
  padding: 0.55rem 1.2rem;
  font-weight: 600;
}

.stButton > button:hover {
  filter: brightness(0.95);
}

/* ===== TABS ===== */
/* ===== TABS: make them bigger and clearer ===== */
.stTabs [data-baseweb="tab"] {
  font-size: 1.05rem;          /* testo più grande */
  font-weight: 600;            /* più “presenza” */
  padding: 12px 18px;          /* area cliccabile più ampia */
}

.stTabs [aria-selected="true"] {
  font-size: 1.1rem;           /* tab attiva leggermente più grande */
  font-weight: 700;
}

}

.stTabs [aria-selected="true"] {
  color: #e6f4f1 !important;
  border-bottom: 2px solid #2ef2c8;
}

/* ===== PROGRESS ===== */
.stProgress > div > div {
  background-color: #2ef2c8;
}

/* ===== DATAFRAME ===== */
[data-testid="stDataFrame"] {
  background: #0b1412;
  border: 1px solid #1f2f2b;
  border-radius: 8px;
}

/* ===== FOOTER ===== */
.phc-footer {
  margin-top: 3rem;
  padding-top: 1rem;
  border-top: 1px solid #1f2f2b;
  display: flex;
  justify-content: space-between;
  color: #9fbfba;
}

.phc-footer a {
  color: #9fbfba;
  text-decoration: none;
}

.phc-footer a:hover {
  text-decoration: underline;
}

</style>

"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------
# Session state
# ----------------------------
if "records" not in st.session_state:
    st.session_state.records = []

if "last_image" not in st.session_state:
    st.session_state.last_image = None

if "image_protocol_confirmed" not in st.session_state:
    st.session_state.image_protocol_confirmed = False

if "last_eval" not in st.session_state:
    # will store preview output (not saved yet)
    st.session_state.last_eval = None


# ----------------------------
# Utilities
# ----------------------------
def generate_sample_id() -> str:
    year = datetime.now().year
    n = len(st.session_state.records) + 1
    return f"PHC_{year}_{n:06d}"


def export_excel(rows):
    df = pd.DataFrame(rows)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="HybridAI")
        wb = writer.book
        ws = writer.sheets["HybridAI"]
        header_fmt = wb.add_format({"bold": True, "bg_color": "#2ef2c8"})
        for i, c in enumerate(df.columns):
            ws.write(0, i, c, header_fmt)
            ws.set_column(i, i, max(12, min(42, len(str(c)) + 4)))
    out.seek(0)
    return out.read()


# ----------------------------
# Physiology: rule score + prior probs
# ----------------------------
def physio_rule_score(p):
    score = 0
    reasons = []

    if p["fvfm"] >= 0.8:
        score += 2
    else:
        reasons.append("Low Fv/Fm → possible photoinhibition / reduced PSII efficiency")

    if p["chltot"] >= 1.5:
        score += 2
    else:
        reasons.append("Low Chl TOT → reduced pigment content / chlorosis risk")

    if p["cartot"] >= 0.5:
        score += 2
    else:
        reasons.append("Low CAR TOT → reduced photoprotection capacity")

    if p["spad"] >= 40:
        score += 2
    else:
        reasons.append("Low SPAD → reduced relative chlorophyll (chlorosis indicator)")

    if p["qp"] >= 0.7:
        score += 2
    else:
        reasons.append("Low qp → reduced photochemical quenching (ETR limitation)")

    if 0.3 <= p["qn"] <= 0.7:
        score += 2
    else:
        if p["qn"] > 0.7:
            reasons.append("High qN → strong NPQ activation (stress/energy dissipation)")
        else:
            reasons.append("Very low qN → weak photoprotection response (context-dependent)")

    if score >= 10:
        klass = "Healthy"
    elif 6 <= score <= 9:
        klass = "Moderate stress"
    else:
        klass = "High stress"

    return score, klass, reasons


def physio_prior_probs(score):
    # score 0..12 -> smooth prior over classes [Healthy, Moderate, High]
    s = float(score)
    p_healthy = 1 / (1 + np.exp(-(s - 9.5)))
    p_high = 1 / (1 + np.exp((s - 4.5)))
    p_moderate = max(0.0, 1.0 - (p_healthy + p_high))
    p_moderate += np.exp(-0.5 * ((s - 7.0) / 1.6) ** 2) * 0.25

    vec = np.array([p_healthy, p_moderate, p_high], dtype=float)
    vec = np.clip(vec, 1e-6, None)
    vec = vec / vec.sum()
    return vec


# ----------------------------
# Image: features + visual risk
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_efficientnet_feature_extractor():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.eval()
    model.classifier = torch.nn.Identity()  # feature extractor
    preprocess = weights.transforms()
    return model, preprocess


def image_features_torch(img: Image.Image):
    model, preprocess = load_efficientnet_feature_extractor()
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(x)  # [1, 1280]
    v = feat.squeeze(0).cpu().numpy().astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    # proxy visual risk (placeholder until trained model exists)
    risk = float(np.clip(1.0 - (v.std() * 3.0), 0.0, 1.0))
    return v, risk


def image_features_fallback(img: Image.Image):
    arr = np.array(img.convert("RGB")).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness = (0.299 * r + 0.587 * g + 0.114 * b)

    g_mean = float(g.mean() / 255.0)
    bright_mean = float(brightness.mean() / 255.0)

    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / (mx + 1e-6)
    sat_mean = float(sat.mean())

    tex = float(np.clip(brightness.var() / (255.0 ** 2), 0.0, 1.0))

    v = np.array([g_mean, bright_mean, sat_mean, tex], dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)

    # handcrafted visual risk
    risk = 0.0
    if g_mean < 0.45:
        risk += 0.35
    if bright_mean < 0.40:
        risk += 0.25
    if sat_mean < 0.18:
        risk += 0.20
    if tex > 0.12:
        risk += 0.15
    risk = float(np.clip(risk, 0.0, 1.0))

    return v, risk


def image_visual_probs(risk):
    # [Healthy, Moderate, High]
    p_high = risk ** 1.2
    p_healthy = (1.0 - risk) ** 1.2
    p_moderate = 1.0 - (p_high + p_healthy)
    vec = np.array([p_healthy, max(0.0, p_moderate), p_high], dtype=float)
    vec = np.clip(vec, 1e-6, None)
    vec = vec / vec.sum()
    return vec


# ----------------------------
# Fusion + explainability
# ----------------------------
def fuse_probs(phys_prior, vis_probs, w_phys=0.70):
    eps = 1e-8
    lp = w_phys * np.log(phys_prior + eps) + (1 - w_phys) * np.log(vis_probs + eps)
    p = np.exp(lp - lp.max())
    p = p / p.sum()
    return p


def explain_prediction(fused_probs, img_risk, phys_score, phys_reasons):
    bullets = []
    if phys_score >= 10:
        bullets.append("Physiology strongly consistent with an optimal PSII functional state (high rule-score).")
    elif phys_score <= 5:
        bullets.append("Physiology indicates likely functional impairment / stress (low rule-score).")
    else:
        bullets.append("Physiology suggests moderate deviation from the optimal state (mid rule-score).")

    if phys_reasons:
        bullets.append("Key physiological flags: " + "; ".join(phys_reasons[:2]) + ("…" if len(phys_reasons) > 2 else ""))

    if img_risk < 0.33:
        bullets.append("Leaf image visually consistent with healthy pigmentation (low visual risk).")
    elif img_risk < 0.66:
        bullets.append("Leaf image shows some visual deviations (medium visual risk).")
    else:
        bullets.append("Leaf image shows strong visual stress cues (high visual risk).")

    conf = float(np.max(fused_probs))
    if conf >= 0.80:
        bullets.append("High confidence (signals agree).")
    elif conf >= 0.60:
        bullets.append("Medium confidence (partial agreement).")
    else:
        bullets.append("Low confidence (mixed signals; consider re-checking measurements and image conditions).")

    return " ".join(bullets)


# ----------------------------
# Header / hero
# ----------------------------
st.markdown(
    f"""
    <div class="phc-hero">
      <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
        <div style="font-size:44px;">🌿</div>
        <div>
          <div style="font-size:2.1rem; font-weight:820; line-height:1.05;">Plant Health Checker – Hybrid AI</div>
          <div class="small-muted">Distribution-ready data collection: image + physiology (no database).</div>
          <div class="small-muted">Torch: <b>{"✅ available" if TORCH_AVAILABLE else "⚪ not installed (fallback vision features)"}</b></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")

tabs = st.tabs(["🧬 Image Acquisition", "🧪 Evaluate (Preview)", "📦 Records & Export"])


# ============================
# TAB 1: Image Acquisition
# ============================
with tabs[0]:
    st.markdown('<div class="phc-card">', unsafe_allow_html=True)
    st.subheader("Leaf image acquisition")

    st.markdown("""
**Leaf image acquisition protocol (recommended):**
- Single, fully expanded leaf
- Neutral background
- Diffuse natural light
- No flash, no strong shadows
- Camera perpendicular to leaf surface
""")

    st.session_state.image_protocol_confirmed = st.checkbox(
        "I confirm that the image was acquired following the recommended conditions",
        value=st.session_state.image_protocol_confirmed
    )

    up = st.file_uploader("Upload leaf image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up).convert("RGB")
        st.session_state.last_image = img
        st.image(img, use_container_width=True, caption="Uploaded leaf image")

        # show visual risk preview
        if TORCH_AVAILABLE:
            _, img_risk = image_features_torch(img)
            method = "EfficientNet-B0 features"
        else:
            _, img_risk = image_features_fallback(img)
            method = "Fallback color/texture heuristics"

        st.markdown(f"**Image analysis method:** {method}")
        st.markdown("**Visual stress index (proxy):**")
        st.progress(int(img_risk * 100))
        st.caption(f"{img_risk:.2f}  (0 = healthy-looking, 1 = strong visual stress cues)")
    else:
        st.info("Upload an image to proceed with hybrid evaluation.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "Note: the visual stress index is an evidence signal for fusion; it is not a trained disease classifier (yet)."
    )


# ============================
# TAB 2: Evaluate (Preview) – no saving unless explicit
# ============================
with tabs[1]:
    left, right = st.columns([0.58, 0.42], gap="large")

    with left:
        st.markdown('<div class="phc-card">', unsafe_allow_html=True)
        st.subheader("Physiological inputs (for evaluation)")

        c1, c2 = st.columns(2)
        with c1:
            sample_name = st.text_input("Sample Name", value="")
        with c2:
            species = st.text_input("Species (optional, for your records)", value="")

        d1, d2, d3 = st.columns(3)
        with d1:
            fvfm = st.number_input(
                "Fv/Fm",
                min_value=0.0, max_value=1.0, value=0.80, step=0.01, format="%.2f",
                help="Maximum quantum efficiency of PSII (dark-adapted). Typical healthy leaves ~0.80–0.83."
            )
            chltot = st.number_input(
                "Chl TOT (per leaf area)",
                min_value=0.0, value=1.50, step=0.01, format="%.2f",
                help="Total chlorophyll per leaf area (units depend on your method). Lower values may indicate chlorosis."
            )
        with d2:
            cartot = st.number_input(
                "CAR TOT (per leaf area)",
                min_value=0.0, value=0.50, step=0.01, format="%.2f",
                help="Total carotenoids per leaf area. Related to photoprotection and stress response."
            )
            spad = st.number_input(
                "SPAD",
                min_value=0.0, value=40.0, step=0.5, format="%.1f",
                help="Relative chlorophyll content (SPAD units). Lower values can indicate pigment reduction."
            )
        with d3:
            qp = st.number_input(
                "qp",
                min_value=0.0, max_value=1.0, value=0.70, step=0.01, format="%.2f",
                help="Photochemical quenching. Lower values may indicate limitations in electron transport."
            )
            qn = st.number_input(
                "qN",
                min_value=0.0, max_value=1.0, value=0.50, step=0.01, format="%.2f",
                help="Non-photochemical quenching. High values often reflect increased energy dissipation under stress."
            )

        st.caption(
            "Bootstrapping note: physiology thresholds define a research-grounded prior; fused with image evidence for hybrid prediction."
        )

        if st.button("Evaluate (Preview)"):
            # physiology
            p = dict(fvfm=fvfm, chltot=chltot, cartot=cartot, spad=spad, qp=qp, qn=qn)
            phys_score, phys_class, phys_reasons = physio_rule_score(p)
            phys_prior = physio_prior_probs(phys_score)

            # image evidence
            if st.session_state.last_image is None:
                img_risk = 0.5
                vis_probs = np.array([1/3, 1/3, 1/3], dtype=float)
                img_method = "No image"
            else:
                img = st.session_state.last_image
                if TORCH_AVAILABLE:
                    _, img_risk = image_features_torch(img)
                    img_method = "EfficientNet-B0 features"
                else:
                    _, img_risk = image_features_fallback(img)
                    img_method = "Fallback color/texture heuristics"
                vis_probs = image_visual_probs(img_risk)

            fused = fuse_probs(phys_prior, vis_probs, w_phys=0.70)
            classes = ["Healthy", "Moderate stress", "High stress"]
            pred_idx = int(np.argmax(fused))
            pred_class = classes[pred_idx]
            conf = float(fused[pred_idx])
            explanation = explain_prediction(fused, img_risk, phys_score, phys_reasons)

            # image metadata for preview/save
            image_metadata = {
                "ImageUploaded": st.session_state.last_image is not None,
                "ImageProtocolConfirmed": bool(st.session_state.image_protocol_confirmed),
                "ImageAnalysisMethod": img_method,
                "TorchUsed": bool(TORCH_AVAILABLE),
                "VisualRiskIndex": round(float(img_risk), 4),
            }

            st.session_state.last_eval = {
                "inputs": {
                    "Sample Name": sample_name.strip(),
                    "Species (optional)": species.strip(),
                    "Fv/Fm": fvfm,
                    "Chl TOT": chltot,
                    "CAR TOT": cartot,
                    "SPAD": spad,
                    "qp": qp,
                    "qN": qn,
                },
                "physio": {"PhysioScore": phys_score, "PhysioClass": phys_class},
                "prediction": {
                    "Prediction": pred_class,
                    "Confidence": round(conf, 4),
                    "Explanation": explanation,
                    "Probs": [round(x, 4) for x in fused.tolist()]
                },
                "image_metadata": image_metadata
            }

            st.success("Preview computed. Review results on the right, then Save if valid.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="phc-card">', unsafe_allow_html=True)
        st.subheader("Preview output (not saved yet)")

        if st.session_state.last_eval is None:
            st.info("Click **Evaluate (Preview)** to compute a prediction.")
        else:
            ev = st.session_state.last_eval
            pred = ev["prediction"]
            md = ev["image_metadata"]
            phys = ev["physio"]

            st.markdown(f"### Prediction: **{pred['Prediction']}**")
            st.markdown("**Confidence:**")
            st.progress(int(float(pred["Confidence"]) * 100))
            st.caption(str(pred["Confidence"]))

            with st.expander("Probability breakdown (Healthy / Moderate / High)"):
                st.write(pred["Probs"])

            st.markdown("**Explainability:**")
            st.write(pred["Explanation"])

            st.markdown("---")
            st.markdown("**Physiology score:**")
            st.write({"PhysioScore": phys["PhysioScore"], "PhysioClass": phys["PhysioClass"]})

            st.markdown("**Image metadata:**")
            st.write(md)

            # SAVE (explicit)
            if st.button("Save record"):
                # gating rules
                if md["ImageUploaded"] is False:
                    st.warning("Cannot save: no image uploaded. Upload an image first (tab: Image Acquisition).")
                elif md["ImageProtocolConfirmed"] is False:
                    st.warning("Cannot save: please confirm the image acquisition protocol checkbox before saving.")
                else:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    sample_id = generate_sample_id()

                    row = {
                        "Sample_ID": sample_id,
                        "Timestamp": ts,

                        "Sample Name": ev["inputs"]["Sample Name"],
                        "Species (optional)": ev["inputs"]["Species (optional)"],

                        "Fv/Fm": ev["inputs"]["Fv/Fm"],
                        "Chl TOT": ev["inputs"]["Chl TOT"],
                        "CAR TOT": ev["inputs"]["CAR TOT"],
                        "SPAD": ev["inputs"]["SPAD"],
                        "qp": ev["inputs"]["qp"],
                        "qN": ev["inputs"]["qN"],

                        "PhysioScore": phys["PhysioScore"],

                        "Prediction": pred["Prediction"],
                        "Confidence": pred["Confidence"],
                        "Explanation": pred["Explanation"],

                        "ImageUploaded": md["ImageUploaded"],
                        "ImageProtocolConfirmed": md["ImageProtocolConfirmed"],
                        "ImageAnalysisMethod": md["ImageAnalysisMethod"],
                        "TorchUsed": md["TorchUsed"],
                        "VisualRiskIndex": md["VisualRiskIndex"],
                    }

                    st.session_state.records.append(row)
                    st.success(f"Saved ✅  Sample_ID: {sample_id}")

        st.caption(
            "This tool provides a physiology-based stress assessment for research purposes only. "
            "It is not a diagnostic system."
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ============================
# TAB 3: Records & Export
# ============================
with tabs[2]:
    st.markdown('<div class="phc-card">', unsafe_allow_html=True)
    st.subheader("Records & export")

    if not st.session_state.records:
        st.info("No saved records yet. Compute a preview and click **Save record**.")
    else:
        df = pd.DataFrame(st.session_state.records)
        st.dataframe(df, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns([0.42, 0.28, 0.30])
        with col1:
            xlsx = export_excel(st.session_state.records)
            st.download_button(
                "Download Excel (.xlsx)",
                data=xlsx,
                file_name="plant_health_hybrid_records.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with col2:
            if st.button("Reset records"):
                st.session_state.records = []
                st.session_state.last_eval = None
                st.success("Cleared ✅")
        with col3:
            st.caption(f"Total records: {len(st.session_state.records)}")

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """
    <div class="phc-footer">
      <div class="small-muted">
        <strong>Contacts</strong><br/>
        <a href="mailto:giuseppemuscari.gm@gmail.com">giuseppemuscari.gm@gmail.com</a> ·
        <a href="#" target="_blank">LinkedIn</a> ·
        <a href="#" target="_blank">Instagram</a>
      </div>
      <div class="small-muted" style="text-align:right;">
        ©2025 <strong>Giuseppe Muscari Tomajoli</strong><br/>
        <span>Plant Health Checker – Hybrid AI</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
