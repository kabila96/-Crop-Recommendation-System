import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "crop_recommendation_outputs"
RF_MODEL_PATH = OUTPUT_DIR / "random_forest_crop_model.joblib"
ADA_MODEL_PATH = OUTPUT_DIR / "adaboost_crop_model.joblib"
RANGE_PATH = OUTPUT_DIR / "feature_ranges.json"
METRICS_PATHS = [OUTPUT_DIR / "quick_metrics.json", BASE_DIR / "quick_metrics.json"]
ADVISORY_PATH = OUTPUT_DIR / "crop_advisory_notes.json"

ABOUT_IMG = BASE_DIR / "about.png"
CAROUSEL_IMAGES = [
    BASE_DIR / "carousel1.png",
    BASE_DIR / "carousel2.png",
    BASE_DIR / "carousel4.png",
    BASE_DIR / "b288b3d0-eefb-48ff-8c3c-a0d9389f7715.png",
]

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .block-container {padding-top: 0.8rem; padding-bottom: 1rem; max-width: 1420px;}
    .hero-title {font-size: clamp(2.1rem, 4vw, 3.5rem); font-weight: 900; line-height: 1.04; margin-bottom: 0.35rem; color: #0f172a; word-break: break-word;}
    .hero-subtitle {font-size: 1.02rem; color: #475569; margin-bottom: 0.9rem; max-width: 780px;}
    .section-title {font-size: 1.35rem; font-weight: 800; color: #0f172a; margin-top: 0.2rem; margin-bottom: 0.45rem;}
    .highlight-card, .info-card, .metric-card, .crop-card, .advice-card {
        border: 1px solid rgba(15,23,42,0.08); border-radius: 20px; padding: 1rem 1.05rem;
        box-shadow: 0 8px 20px rgba(15,23,42,0.05); background: rgba(248,250,252,0.96);
    }
    .highlight-card {background: linear-gradient(135deg, rgba(22,101,52,0.08), rgba(2,132,199,0.08));}
    .metric-card {background: linear-gradient(135deg, rgba(236,253,245,0.95), rgba(240,249,255,0.95)); min-height: 122px;}
    .metric-label {font-size: 0.85rem; color: #475569; margin-bottom: 0.25rem;}
    .metric-value {font-size: 1.5rem; font-weight: 800; color: #166534;}
    .metric-note {font-size: 0.92rem; color: #334155; margin-top: 0.3rem;}
    .crop-card {background: linear-gradient(135deg, rgba(240,253,244,0.95), rgba(239,246,255,0.95)); min-height: 150px;}
    .crop-rank {font-size: 0.88rem; color: #475569;}
    .crop-name {font-size: 1.35rem; font-weight: 800; color: #14532d; margin: 0.2rem 0;}
    .crop-prob {font-size: 1.1rem; font-weight: 700; color: #0f172a;}
    .advice-card {background: linear-gradient(135deg, rgba(255,251,235,0.98), rgba(239,246,255,0.96));}
    .small-note {color: #64748b; font-size: 0.9rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 0.45rem; flex-wrap: wrap;}
    .stTabs [data-baseweb="tab"] {border-radius: 12px; padding: 0.48rem 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


@st.cache_resource
def load_models():
    rf_model = joblib.load(RF_MODEL_PATH)
    ada_model = joblib.load(ADA_MODEL_PATH)
    with open(RANGE_PATH, "r", encoding="utf-8") as f:
        ranges = json.load(f)
    return rf_model, ada_model, ranges


def load_metrics():
    for p in METRICS_PATHS:
        if p.exists():
            return load_json(p)
    return {}


def build_input_frame(user_values):
    return pd.DataFrame([user_values])


def predict_top3(model, input_df):
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)[:3]
    return ranked


def metric_box(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def perf_card(title, metric_block):
    if not metric_block:
        st.markdown(
            f"<div class='info-card'><h3>{title}</h3><p>Performance metrics not found.</p></div>",
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"""
        <div class="info-card">
            <div style="font-size:1.1rem;font-weight:800;color:#0f172a;margin-bottom:0.6rem;">{title}</div>
            <div><b>Accuracy:</b> {metric_block.get('accuracy', 0)*100:.2f}%</div>
            <div><b>Macro F1:</b> {metric_block.get('macro_f1', 0)*100:.2f}%</div>
            <div><b>Macro Precision:</b> {metric_block.get('macro_precision', 0)*100:.2f}%</div>
            <div><b>Macro Recall:</b> {metric_block.get('macro_recall', 0)*100:.2f}%</div>
            <div><b>Top-3 Accuracy:</b> {metric_block.get('top3_accuracy', 0)*100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def advisory_notes(crop_name, advisory_map):
    notes = advisory_map.get(crop_name, []) if advisory_map else []
    if not notes:
        return [
            "check rainfall suitability",
            "review soil pH before planting",
            "verify fertilizer needs locally",
        ]
    return notes


if not (RF_MODEL_PATH.exists() and ADA_MODEL_PATH.exists() and RANGE_PATH.exists()):
    st.error("Model files are missing. Run the workflow first to generate models, ranges, and metrics.")
    st.stop()

rf_model, ada_model, ranges = load_models()
metrics = load_metrics()
advisory_map = load_json(ADVISORY_PATH) or {}

with st.sidebar:
    st.markdown("## Model Settings")
    model_name = st.selectbox("Choose prediction model", ["Random Forest", "AdaBoost"])
    st.markdown("---")
    st.markdown(
        """
        **How to use this tool**

        Adjust the farm conditions using the sliders.
        The app returns the best crop, top-3 alternatives,
        model confidence, and simple farmer notes.
        """
    )
    st.markdown("---")
    st.markdown(
        """
        **Field reality check**

        Do not use the model alone.
        Also consider seed availability, pest pressure,
        irrigation access, local extension advice,
        and market demand.
        """
    )

hero_left, hero_right = st.columns([1.18, 1], gap="large")
with hero_left:
    st.markdown('<div class="hero-title">🌾 Crop Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">A machine learning decision-support app for recommending suitable crops based on soil nutrients and environmental conditions.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="highlight-card">
        <b>Created by Powell Andile Ndlovu | Data Analyst</b><br><br>
        This prototype helps farmers and agricultural planners compare soil and weather conditions against a trained crop recommendation model.
        Treat it as decision support, not a substitute for agronomic judgement.
        </div>
        """,
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_box("Models", "2", "Random Forest and AdaBoost benchmarked for crop recommendation")
    with m2:
        metric_box("Input Features", "7", "N, P, K, temperature, humidity, pH, and rainfall")
    with m3:
        metric_box("Primary Goal", "Farmer Support", "Recommend the most suitable crop under current conditions")
with hero_right:
    if ABOUT_IMG.exists():
        st.image(str(ABOUT_IMG), use_container_width=True)

st.markdown("")
gallery_tab, predict_tab, perf_tab, explain_tab, eda_tab = st.tabs(
    ["Overview", "Crop Prediction", "Algorithm Performance", "SHAP & LIME", "Data Analysis"]
)

with gallery_tab:
    st.markdown('<div class="section-title">Project slideshow</div>', unsafe_allow_html=True)
    available_images = [p for p in CAROUSEL_IMAGES if p.exists()]
    if "carousel_index" not in st.session_state:
        st.session_state.carousel_index = 0
    prev_col, center_col, next_col = st.columns([1, 8, 1])
    with prev_col:
        if st.button("◀ Previous", use_container_width=True):
            st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(available_images)
    with next_col:
        if st.button("Next ▶", use_container_width=True):
            st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(available_images)
    if available_images:
        current = available_images[st.session_state.carousel_index]
        with center_col:
            st.image(str(current), use_container_width=True, caption=f"Slide {st.session_state.carousel_index + 1} of {len(available_images)}")
    st.info("Overview of Crop Recomendation App.")

with predict_tab:
    st.markdown('<div class="section-title">Enter farm conditions</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Adjust the sliders to match observed soil and environmental conditions for the field.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        N = st.slider("Nitrogen (N)", float(ranges["N"]["min"]), float(ranges["N"]["max"]), float(ranges["N"]["mean"]))
        P = st.slider("Phosphorus (P)", float(ranges["P"]["min"]), float(ranges["P"]["max"]), float(ranges["P"]["mean"]))
        K = st.slider("Potassium (K)", float(ranges["K"]["min"]), float(ranges["K"]["max"]), float(ranges["K"]["mean"]))
    with c2:
        temperature = st.slider("Temperature (°C)", float(ranges["temperature"]["min"]), float(ranges["temperature"]["max"]), float(ranges["temperature"]["mean"]))
        humidity = st.slider("Humidity (%)", float(ranges["humidity"]["min"]), float(ranges["humidity"]["max"]), float(ranges["humidity"]["mean"]))
    with c3:
        ph = st.slider("Soil pH", float(ranges["ph"]["min"]), float(ranges["ph"]["max"]), float(ranges["ph"]["mean"]))
        rainfall = st.slider("Rainfall (mm)", float(ranges["rainfall"]["min"]), float(ranges["rainfall"]["max"]), float(ranges["rainfall"]["mean"]))

    input_values = {"N": N, "P": P, "K": K, "temperature": temperature, "humidity": humidity, "ph": ph, "rainfall": rainfall}
    input_df = build_input_frame(input_values)
    selected_model = rf_model if model_name == "Random Forest" else ada_model
    predicted_crop = selected_model.predict(input_df)[0]
    ranked_predictions = predict_top3(selected_model, input_df)
    prob_df = pd.DataFrame(ranked_predictions, columns=["Crop", "Probability"])
    prob_df["Probability"] = (prob_df["Probability"] * 100).round(2)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="section-title">Recommended crop</div>', unsafe_allow_html=True)
        st.success(f"Best recommendation: **{predicted_crop}**")
        st.dataframe(input_df, use_container_width=True, hide_index=True)
    with right:
        st.markdown('<div class="section-title">Probability view</div>', unsafe_allow_html=True)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
        st.bar_chart(prob_df.set_index("Crop"))

    st.markdown('<div class="section-title">Top-3 crop cards</div>', unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns(3)
    for col, row, rank in zip([tc1, tc2, tc3], prob_df.to_dict("records"), [1, 2, 3]):
        with col:
            st.markdown(
                f"""
                <div class="crop-card">
                    <div class="crop-rank">Rank #{rank}</div>
                    <div class="crop-name">{row['Crop']}</div>
                    <div class="crop-prob">{row['Probability']:.2f}% match</div>
                    <div class="small-note">Model-based crop suitability score from the selected algorithm.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    adv1, adv2 = st.columns([1.1, 1], gap="large")
    with adv1:
        st.markdown('<div class="section-title">Farmer advisory panel</div>', unsafe_allow_html=True)
        notes = advisory_notes(predicted_crop, advisory_map)
        st.markdown(
            "<div class='advice-card'><b>Plain-language notes for this crop</b><br><br>"
            + "<br>".join([f"• {n}" for n in notes])
            + "</div>",
            unsafe_allow_html=True,
        )
    with adv2:
        st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="info-card">
            The <b>{model_name}</b> model judges the current nutrient, rainfall, humidity, temperature, and pH pattern to be most aligned with <b>{predicted_crop}</b>.
            A high score means the current feature profile resembles historical examples of that crop in the training data.
            </div>
            """,
            unsafe_allow_html=True,
        )

    csv = prob_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download prediction summary (CSV)", data=csv, file_name="crop_recommendation_prediction.csv", mime="text/csv")

with perf_tab:
    st.markdown('<div class="section-title">Algorithm performance</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2, gap="large")
    with p1:
        perf_card("Random Forest", metrics.get("Random Forest", {}))
    with p2:
        perf_card("AdaBoost", metrics.get("AdaBoost", {}))
    perf_img = OUTPUT_DIR / "model_performance_comparison.png"
    if perf_img.exists():
        st.image(str(perf_img), use_container_width=True, caption="Accuracy, Macro F1, and Top-3 Accuracy comparison")

with explain_tab:
    st.markdown('<div class="section-title">SHAP and LIME outputs</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2, gap="large")
    with e1:
        st.markdown("**Random Forest explanations**")
        rf_shap = OUTPUT_DIR / "random_forest_shap_summary_bar.png"
        rf_lime = OUTPUT_DIR / "random_forest_lime_explanation.png"
        if rf_shap.exists():
            st.image(str(rf_shap), use_container_width=True, caption="Random Forest SHAP summary")
        else:
            st.warning("Random Forest SHAP image not found.")
        if rf_lime.exists():
            st.image(str(rf_lime), use_container_width=True, caption="Random Forest LIME explanation")
        else:
            st.warning("Random Forest LIME image not found.")
    with e2:
        st.markdown("**AdaBoost explanations**")
        ada_shap = OUTPUT_DIR / "adaboost_shap_summary_bar.png"
        ada_lime = OUTPUT_DIR / "adaboost_lime_explanation.png"
        if ada_shap.exists():
            st.image(str(ada_shap), use_container_width=True, caption="AdaBoost SHAP summary")
        else:
            st.warning("AdaBoost SHAP image not found.")
        if ada_lime.exists():
            st.image(str(ada_lime), use_container_width=True, caption="AdaBoost LIME explanation")
        else:
            st.warning("AdaBoost LIME image not found.")

with eda_tab:
    st.markdown('<div class="section-title">Data analysis outputs</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">These images come from the workflow output folder and help justify the modeling decisions.</div>', unsafe_allow_html=True)
    eda_images = [
        (OUTPUT_DIR / "01_class_distribution.png", "Crop class distribution"),
        (OUTPUT_DIR / "02_correlation_heatmap.png", "Feature correlation heatmap"),
        (OUTPUT_DIR / "03_crop_feature_profile_heatmap.png", "Crop-wise mean profile heatmap"),
        (OUTPUT_DIR / "random_forest_confusion_matrix.png", "Random Forest confusion matrix"),
        (OUTPUT_DIR / "adaboost_confusion_matrix.png", "AdaBoost confusion matrix"),
        (OUTPUT_DIR / "random_forest_permutation_importance.png", "Random Forest permutation importance"),
        (OUTPUT_DIR / "adaboost_permutation_importance.png", "AdaBoost permutation importance"),
    ]
    cols = st.columns(2)
    idx = 0
    for img_path, caption in eda_images:
        if img_path.exists():
            with cols[idx % 2]:
                st.image(str(img_path), use_container_width=True, caption=caption)
            idx += 1

st.markdown("---")
st.caption("Disclaimer: this app is based on the uploaded dataset and should be validated against local agronomic conditions before use in real farming decisions.")
