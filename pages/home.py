import streamlit as st
from pathlib import Path
from src.data_loader import inspect_dataset
from src.model_training import train_and_evaluate
from config import ARTIFACT_DIR


def render(raw):
    info = inspect_dataset(raw)
    result = st.session_state.get("training_result")

    # Road-inspired hero: illustration is decorative; all displayed figures below
    # come from the currently loaded dataset and fitted model.
    left, right = st.columns([1.05, 1.35], gap="large", vertical_alignment="center")
    with left:
        st.markdown(
            '<div class="ts-kicker">SMART TRAFFIC ANALYTICS</div>'
            '<h1 class="ts-home-title">Traffic<span>Sense</span></h1>'
            '<p class="ts-home-subtitle">Understand traffic. Anticipate congestion.</p>'
            '<p class="ts-home-copy">Machine-learning-powered traffic analysis, '
            'congestion prediction and what-if scenario planning for the '
            'Metro Interstate corridor.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ts-road-chip">🚦 &nbsp; DATA-DRIVEN MOBILITY INTELLIGENCE</div>', unsafe_allow_html=True)
    with right:
        banner = Path(__file__).resolve().parents[1] / "assets" / "road_banner.png"
        if banner.exists():
            st.image(str(banner), use_container_width=True)

    st.markdown('<div class="ts-section-label">DATASET AT A GLANCE</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Dataset records", f"{info['rows']:,}")
    b.metric("Original features", f"{info['columns']}")
    c.metric("Unique timestamps", f"{info['unique_timestamps']:,}")
    d.metric("Exact duplicates", f"{info['exact_duplicates']:,}")

    st.markdown('<div class="ts-section-label">MODEL INTELLIGENCE</div>', unsafe_allow_html=True)
    st.subheader("Model performance")
    if result is None:
        st.info("The dataset is loaded. Train and compare the three regression models to unlock predictions and performance insights.")
        if st.button("Train and evaluate models", type="primary", use_container_width=False):
            with st.spinner("Cleaning data, engineering features and training three models. This may take a few minutes..."):
                try:
                    trained = train_and_evaluate(
                        raw,
                        model_dir=ARTIFACT_DIR / "retrained_experiments"
                    )
                    st.session_state.training_result = trained
                    st.session_state.force_retrain = False
                    st.success(f"Training complete. Best model: {trained['best_model']}")
                    st.rerun()
                except Exception as exc:
                    st.session_state.training_error = str(exc)
                    st.error(f"Training failed: {exc}")
        if st.session_state.get("training_error"):
            st.error(st.session_state.training_error)
    else:
        metrics = result["metrics"]
        best = result["best_model"]
        st.markdown(
            f'<div class="ts-model-banner"><span class="ts-model-dot"></span>'
            f'<div><div class="ts-model-eyebrow">BEST PERFORMING MODEL</div>'
            f'<div class="ts-model-name">{best}</div>'
            f'<div class="ts-model-description">Selected using the lowest chronological holdout MAE</div></div>'
            f'<div class="ts-model-road">🚗</div></div>',
            unsafe_allow_html=True,
        )
        best_row = metrics.loc[metrics.model == best].iloc[0] if "model" in metrics.columns else metrics.iloc[0]
        cols = st.columns(3)
        for col, key, label in zip(cols, ["MAE", "RMSE", "R2"], ["Mean Absolute Error", "Root Mean Squared Error", "R² score"]):
            col.metric(label, f"{best_row[key]:,.3f}")
        st.caption("Model comparison · chronological holdout evaluation")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        meta = result["metadata"]
        st.caption(f"Chronological holdout: {meta.get('train_rows', '—'):,} training rows / {meta.get('test_rows', '—'):,} test rows · cutoff {meta.get('split_cutoff', '—')}")

    st.markdown('<div class="ts-section-label">DATASET DETAILS</div>', unsafe_allow_html=True)
    st.subheader("Dataset snapshot")
    left, right = st.columns(2)
    with left:
        st.markdown(
            f'<div class="ts-detail-card"><div class="ts-detail-label">OBSERVATION PERIOD</div>'
            f'<div class="ts-detail-value">{info["timestamp_start"]}</div>'
            f'<div class="ts-detail-arrow">to</div><div class="ts-detail-value">{info["timestamp_end"]}</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="ts-detail-card"><div class="ts-detail-label">HOURLY TRAFFIC RANGE</div>'
            f'<div class="ts-detail-value">{info["target_min"]:,.0f} – {info["target_max"]:,.0f}</div>'
            f'<div class="ts-detail-arrow">vehicles per hour</div></div>',
            unsafe_allow_html=True,
        )
    st.caption("All statistics are calculated from the currently loaded dataset. Model metrics use the chronological holdout; no demonstration values are used.")
