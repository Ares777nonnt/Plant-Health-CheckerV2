import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Optional Torch
TORCH_AVAILABLE = False
try:
    import torch
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ----------------------------
# Page config + CSS theme
# ----------------------------
st.set_page_config(page_title="Plant Health Checker – Hybrid AI", page_icon="🌿", layout="wide")

CUSTOM_CSS = """
<style>
.stApp { background: linear-gradient(135deg, #001a17 0%, #0a3d35 100%); color: #eafffb; }
.block-container { padding-top: 2rem; }
h1,h2,h3,h4 { color: #eafffb; }
.phc-card {
  background: rgba(0, 0, 0, 0.22);
  border: 1px solid rgba(140, 255, 230, 0.18);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.18);
}
.phc-hero {
  border-radius: 22px;
  padding: 22px;
  background: radial-gradient(circle at 15% 20%, rgba(0, 255, 200, 0.12), transparent 45%),
              radial-gradient(circle at 80% 10%, rgba(0, 200, 255, 0.10), transparent 40%),
              rgba(0, 0, 0, 0.22);
  border: 1px solid rgba(140, 255, 230, 0.18);
  box-shadow: 0 10px 40px rgba(0,0,0,0.20);
}
.stButton > button {
  border: 0; border-radius: 14px; padding: 0.7rem 1.1rem; font-weight: 650;
  color: #001a17;
  background: linear-gradient(90deg, #2ef2c8 0%, #00d1ff 100%);
  box-shadow: 0 10px 22px rgba(0,0,0,0.25);
}
.stButton > button:hover { filter: brightness(1.05); transform: translateY(-1px); }
.stButton > button:active { filter: brightness(0.98); transform: translateY(0px); }

.stTextInput input, .stNumberInput input {
  border-radius: 12px !important;
  background: rgba(0,0,0,0.25) !important;
  border: 1px solid rgba(140,255,230,0.22) !important;
  color: #eafffb !important;
}
.phc-footer {
  margin-top: 2.2rem; padding: 16px 18px; border-radius: 18px;
  background: rgba(0, 0, 0, 0.18);
  border: 1px solid rgba(140, 255, 230, 0.18);
  display: flex; justify-content: space-between; align-items: center;
  gap: 14px; flex-wrap: wrap;
}
.phc-footer a { color: #2ef2c8; text-decoration: none; font-weight: 650; }
.phc-footer a:hover { text-decoration: underline; }
.small-muted { opacity: 0.85; font-size: 0.92rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------
# Session state
# ----------------------------
if "records" not in st.session_state:
    st.session_state.records = []


# ----------------------------
# Helpers: physiology rule score (your thresholds)
# ----------------------------
def physio_rule_score(p):
    score = 0
    reasons = []

    if p["fvfm"] >= 0.8:
        score += 2
    else:
        reasons.append("Low Fv/Fm → possible photoinhibition / PSII inefficiency")

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
        reasons.append("Low qp → reduced photochemical quenching (electron transport limitation)")

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
    # Map score (0..12) -> prior probabilities over classes
    # (handcrafted but smooth; later replace with trained MLP)
    # Healthy increases sharply at high scores, High stress at low scores.
    s = float(score)

    # Simple piecewise-ish logistic curves
    p_healthy = 1 / (1 + np.exp(-(s - 9.5)))
    p_high = 1 / (1 + np.exp((s - 4.5)))
    # moderate gets the leftover + bump around mid-scores
    p_moderate = max(0.0, 1.0 - (p_healthy + p_high))
    p_moderate += np.exp(-0.5 * ((s - 7.0) / 1.6) ** 2) * 0.25

    vec = np.array([p_healthy, p_moderate, p_high], dtype=float)
    vec = np.clip(vec, 1e-6, None)
    vec = vec / vec.sum()
    return vec  # [Healthy, Moderate, High]


# ----------------------------
# Helpers: image features (Torch or fallback)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_efficientnet_feature_extractor():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.eval()
    # Remove the classifier head; keep features
    model.classifier = torch.nn.Identity()
    preprocess = weights.transforms()
    return model, preprocess


def image_features_torch(img: Image.Image):
    model, preprocess = load_efficientnet_feature_extractor()
    x = preprocess(img).unsqueeze(0)  # [1,3,H,W]
    with torch.no_grad():
        feat = model(x)  # [1, 1280]
    v = feat.squeeze(0).cpu().numpy().astype(np.float32)
    # Normalize
    v = v / (np.linalg.norm(v) + 1e-8)
    # Compute a simple visual risk index from feature magnitude distribution (proxy)
    risk = float(np.clip(1.0 - (v.std() * 3.0), 0.0, 1.0))
    return v, risk


def image_features_fallback(img: Image.Image):
    # Lightweight visual features: green mean, brightness mean, saturation proxy, texture proxy
    arr = np.array(img.convert("RGB")).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness = (0.299 * r + 0.587 * g + 0.114 * b)
    g_mean = float(g.mean() / 255.0)
    bright_mean = float(brightness.mean() / 255.0)

    # Saturation proxy (simple)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / (mx + 1e-6)
    sat_mean = float(sat.mean())

    # Texture proxy: variance of brightness
    tex = float(np.clip(brightness.var() / (255.0 ** 2), 0.0, 1.0))

    # pack into a small vector
    v = np.array([g_mean, bright_mean, sat_mean, tex], dtype=np.float32)

    # Visual risk: low green/brightness or very low saturation can indicate yellow/brown/dark patterns
    # (handcrafted heuristic; replace with trained CNN later)
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

    # normalize vector
    v = v / (np.linalg.norm(v) + 1e-8)
    return v, risk


def image_visual_probs(risk):
    # Map visual risk [0..1] to class probs (Healthy, Moderate, High)
    # risk low => healthy; risk high => high stress
    p_high = risk ** 1.2
    p_healthy = (1.0 - risk) ** 1.2
    p_moderate = 1.0 - (p_high + p_healthy)
    vec = np.array([p_healthy, max(0.0, p_moderate), p_high], dtype=float)
    vec = np.clip(vec, 1e-6, None)
    vec = vec / vec.sum()
    return vec


# ----------------------------
# Fusion: combine physiology prior + visual likelihood
# ----------------------------
def fuse_probs(phys_prior, vis_probs, w_phys=0.65):
    # Weighted log-space fusion (stable)
    # final ∝ exp( w*log(phys) + (1-w)*log(vis) )
    eps = 1e-8
    lp = w_phys * np.log(phys_prior + eps) + (1 - w_phys) * np.log(vis_probs + eps)
    p = np.exp(lp - lp.max())
    p = p / p.sum()
    return p


def explain_prediction(p, img_risk, phys_score, phys_reasons):
    # Build a compact explanation emphasizing physiology + note on visual signal
    bullets = []
    if phys_score >= 10:
        bullets.append("Physiology strongly consistent with an optimal PSII state (high rule-score).")
    elif phys_score <= 5:
        bullets.append("Physiology indicates likely functional impairment / stress (low rule-score).")
    else:
        bullets.append("Physiology suggests moderate deviation from optimal state (mid rule-score).")

    # add top physiological reasons (if any)
    if phys_reasons:
        bullets.append("Key physiological flags: " + "; ".join(phys_reasons[:2]) + ("…" if len(phys_reasons) > 2 else ""))

    # visual note
    if img_risk < 0.33:
        bullets.append("Leaf image looks visually consistent with healthy pigmentation (low visual risk).")
    elif img_risk < 0.66:
        bullets.append("Leaf image shows some visual deviations (medium visual risk).")
    else:
        bullets.append("Leaf image shows strong visual stress cues (high visual risk).")

    # confidence interpretation
    conf = float(np.max(p))
    if conf >= 0.80:
        bullets.append("High confidence (strong agreement between signals).")
    elif conf >= 0.60:
        bullets.append("Medium confidence (partial agreement between signals).")
    else:
        bullets.append("Low confidence (signals are mixed; consider re-checking measurements and image conditions).")

    return " ".join(bullets)


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
            ws.set_column(i, i, max(12, min(36, len(c) + 4)))
    out.seek(0)
    return out.read()


# ----------------------------
# UI: Hero
# ----------------------------
st.markdown(
    f"""
    <div class="phc-hero">
      <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
        <div style="font-size:44px;">🌿</div>
        <div>
          <div style="font-size:2.1rem; font-weight:820; line-height:1.05;">Plant Health Checker – Hybrid AI</div>
          <div class="small-muted">AI fusion: leaf-image features + physiological parameters (no database).</div>
          <div class="small-muted">Torch: <b>{"✅ available" if TORCH_AVAILABLE else "⚪ not installed (fallback vision features)"}</b></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")


tabs = st.tabs(["🧬 Hybrid AI Analysis", "🧪 Physiological Input & Fusion", "📦 Sample Records & Export"])

# Shared state across tabs
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_image" not in st.session_state:
    st.session_state.last_image = None


# ----------------------------
# TAB 1: image upload + visual branch
# ----------------------------
with tabs[0]:
    st.markdown('<div class="phc-card">', unsafe_allow_html=True)
    st.subheader("Hybrid AI – Image branch")

    col1, col2 = st.columns([0.55, 0.45], gap="large")
    with col1:
        up = st.file_uploader("Upload leaf image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
        if up:
            img = Image.open(up).convert("RGB")
            st.session_state.last_image = img
            st.image(img, use_container_width=True, caption="Uploaded leaf image")

    with col2:
        if st.session_state.last_image is None:
            st.info("Upload an image to extract visual features (used in the fusion step).")
        else:
            img = st.session_state.last_image
            if TORCH_AVAILABLE:
                feat, risk = image_features_torch(img)
                method = "EfficientNet-B0 feature extractor"
                dim = feat.shape[0]
            else:
                feat, risk = image_features_fallback(img)
                method = "Fallback vision features (green/brightness/texture proxies)"
                dim = feat.shape[0]

            st.markdown(f"**Visual method:** {method}")
            st.markdown(f"**Feature dim:** `{dim}`")
            st.markdown("**Visual stress index:**")
            st.progress(int(risk * 100))
            st.caption(f"Visual risk ≈ {risk:.2f} (0 = healthy-looking, 1 = strong visual stress cues)")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.caption(
        "Note: this branch does not output a final diagnosis by itself — it provides visual evidence to be fused with physiology."
    )


# ----------------------------
# TAB 2: physiology input + fusion prediction
# ----------------------------
with tabs[1]:
    left, right = st.columns([0.58, 0.42], gap="large")

    with left:
        st.markdown('<div class="phc-card">', unsafe_allow_html=True)
        st.subheader("Physiological inputs")

        c1, c2 = st.columns(2)
        with c1:
            sample_name = st.text_input("Sample Name", value="")
        with c2:
            # species kept as free text for logging only (no DB)
            species = st.text_input("Species (optional, for your records)", value="")

        d1, d2, d3 = st.columns(3)
        with d1:
            fvfm = st.number_input("Fv/Fm", min_value=0.0, max_value=1.0, value=0.80, step=0.01, format="%.2f")
            chltot = st.number_input("Chl TOT (per leaf area)", min_value=0.0, value=1.50, step=0.01, format="%.2f")
        with d2:
            cartot = st.number_input("CAR TOT (per leaf area)", min_value=0.0, value=0.50, step=0.01, format="%.2f")
            spad = st.number_input("SPAD", min_value=0.0, value=40.0, step=0.5, format="%.1f")
        with d3:
            qp = st.number_input("qp", min_value=0.0, max_value=1.0, value=0.70, step=0.01, format="%.2f")
            qn = st.number_input("qN", min_value=0.0, max_value=1.0, value=0.50, step=0.01, format="%.2f")

        st.caption(
            "Rule-based pseudo-labeling (bootstrapping): each threshold contributes +2 points (max 12). "
            "This provides a physiology-grounded prior that is fused with the image evidence."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="phc-card">', unsafe_allow_html=True)
        st.subheader("AI Fusion prediction")

        if st.session_state.last_image is None:
            st.warning("Upload an image first (tab: Hybrid AI Analysis) for true hybrid fusion.")
            st.caption("You can still evaluate physiology-only, but the fusion confidence will be limited.")

        p = dict(fvfm=fvfm, chltot=chltot, cartot=cartot, spad=spad, qp=qp, qn=qn)
        phys_score, phys_class, phys_reasons = physio_rule_score(p)
        phys_prior = physio_prior_probs(phys_score)

        # visual probabilities
        if st.session_state.last_image is None:
            # neutral visual evidence
            vis_probs = np.array([1/3, 1/3, 1/3], dtype=float)
            img_risk = 0.5
        else:
            img = st.session_state.last_image
            if TORCH_AVAILABLE:
                _, img_risk = image_features_torch(img)
            else:
                _, img_risk = image_features_fallback(img)
            vis_probs = image_visual_probs(img_risk)

        # Fuse
        fused = fuse_probs(phys_prior, vis_probs, w_phys=0.70)

        classes = ["Healthy", "Moderate stress", "High stress"]
        pred_idx = int(np.argmax(fused))
        pred_class = classes[pred_idx]
        conf = float(fused[pred_idx])

        st.markdown(f"### Prediction: **{pred_class}**")
        st.markdown("**Confidence:**")
        st.progress(int(conf * 100))
        st.caption(f"{conf:.2f}")

        with st.expander("See probability breakdown"):
            st.write({
                "Physiology prior (H/M/High)": [round(x, 3) for x in phys_prior.tolist()],
                "Visual evidence (H/M/High)": [round(x, 3) for x in vis_probs.tolist()],
                "Fused posterior (H/M/High)": [round(x, 3) for x in fused.tolist()],
            })

        explanation = explain_prediction(fused, img_risk, phys_score, phys_reasons)
        st.markdown("**Explainability:**")
        st.write(explanation)

        # Save record
        if st.button("Evaluate & Save record"):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = {
                "Timestamp": ts,
                "Sample Name": sample_name.strip(),
                "Species": species.strip(),
                "Fv/Fm": fvfm,
                "Chl TOT": chltot,
                "CAR TOT": cartot,
                "SPAD": spad,
                "qp": qp,
                "qN": qn,
                "Physio score": phys_score,
                "Prediction": pred_class,
                "Confidence": conf,
                "Explanation": explanation,
                "Torch": TORCH_AVAILABLE
            }
            st.session_state.records.append(row)
            st.success("Saved ✅")

        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# TAB 3: records + export
# ----------------------------
with tabs[2]:
    st.markdown('<div class="phc-card">', unsafe_allow_html=True)
    st.subheader("Sample Records & Export")

    if not st.session_state.records:
        st.info("No records saved yet. Run “Evaluate & Save record” in the Fusion tab.")
    else:
        df = pd.DataFrame(st.session_state.records)
        st.dataframe(df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns([0.45, 0.55])
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
                st.success("Cleared ✅")

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
