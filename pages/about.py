import streamlit as st

def render(raw):
    st.markdown('<div class="ts-page-kicker">PROJECT & METHODOLOGY</div><div class="ts-page-intro">Understand the dataset, model pipeline, evaluation approach and project assumptions.</div>', unsafe_allow_html=True)
    st.title("About & Methodology")
    st.markdown("""
### Project overview
TrafficSense predicts hourly traffic volume from historical traffic observations, weather conditions, holidays and timestamp-derived features.

### Dataset
The supplied Metro Interstate Traffic Volume dataset contains the target `traffic_volume`, timestamp `date_time`, holiday labels, temperature, rainfall, snowfall, cloud coverage and two weather descriptors. Temperature is stored in Kelvin in the source data.

### Data preparation
- Exact duplicate rows are removed; repeated timestamps with distinct observations are retained.
- Invalid timestamps and missing target values are excluded.
- Invalid negative precipitation/cloud values are treated as missing; cloud coverage is constrained to 0–100.
- Zero temperature is treated as missing because it is not a plausible Kelvin observation for this dataset.
- Missing numeric and categorical feature values are imputed inside the training pipeline.
- Rainfall and snowfall are transformed using `log1p` to reduce skew without deleting observed extremes.

### Feature engineering
Hour, weekday, day, month, day-of-year, weekend and holiday indicators are created. Cyclical sine/cosine features represent hour, weekday and month. Weather categories and holiday names are one-hot encoded.

### Training and model selection
Random Forest, Gradient Boosting and HistGradientBoosting regressors are trained and evaluated using a chronological split. All rows sharing a timestamp are kept on the same side of the split. The best model is selected by lowest holdout MAE, with RMSE as a tie-breaker. The fitted preprocessing and regression pipeline is persisted together.

### Evaluation metrics
- **MAE:** average absolute prediction error in vehicles per hour.
- **RMSE:** square-root average squared error, giving greater weight to large misses.
- **R²:** proportion of target variance explained relative to a mean baseline; it can be negative on a difficult holdout.

### Scenario and holiday assumptions
Scenario temperature is entered in Celsius and converted to Kelvin. Holiday detection first uses date-level labels found in the uploaded history; for other dates it uses the US–Minnesota holiday calendar when the optional `holidays` dependency is available. The weather description is populated using the most frequent description associated with the selected main weather category. Traffic categories use one-third and two-thirds quantiles of the training target distribution, so thresholds are derived from the training data rather than invented fixed cutoffs.

### Limitations
This is an observational prediction model, not a causal traffic simulator. Weather inputs are assumed to be known for the selected scenario. Predictions outside the historical time/weather range may be less reliable. A holiday calendar cannot identify every local event or unusual closure. The source dataset has repeated timestamps with multiple observations; these are preserved unless the complete row is duplicated.
""")
