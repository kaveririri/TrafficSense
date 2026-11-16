# TrafficSense 2.0
### Intelligent Traffic Prediction & Scenario Analysis

TrafficSense is a machine-learning-powered traffic analytics platform designed to predict traffic volume, explore historical patterns, and simulate different weather and environmental scenarios.

## Features

- **Traffic Prediction:** Predict traffic volume using weather, date and time inputs.
- **Analytics Dashboard:** Explore historical traffic patterns and trends.
- **Scenario Simulator:** Evaluate the impact of different weather conditions.
- **Scenario Comparison:** Compare baseline and alternative traffic scenarios.
- **Weather Visualization:** Animated weather scenes reflecting selected conditions.
- **Model Lab:** Explore model performance and evaluation metrics.
- **Automatic Holiday Detection:** Detects holidays recorded for matching dates in the dataset.

## Machine Learning Model

Selected model: **HistGradientBoosting**

| Metric | Result |
|---|---:|
| MAE | 263.43 |
| RMSE | 452.73 |
| R² Score | 0.94697 |

The model uses 16 input features, including temperature, rainfall, snowfall, cloud coverage, date/time features, holidays and weather descriptions.

## Dataset

**Metro Interstate Traffic Volume Dataset**

- Original records: 48,204
- Cleaned records: 48,187
- Target variable: `traffic_volume`

The dataset contains historical traffic observations from the I-94 interstate.

**Note:** This application does not provide live traffic data or GPS tracking. Holiday detection is limited to information recorded in the dataset and does not cover every Indian festival.

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly
- Joblib
- Holidays

## Project Structure

```text
TrafficSense/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── artifacts/
├── data/
├── models/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── prediction.py
│   ├── scenario_analysis.py
│   └── visualization.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/kaveririri/TrafficSense.git
cd TrafficSense
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Model Artifacts

- `models/traffic_pipeline.joblib` — Saved preprocessing and prediction pipeline.
- `artifacts/model_metadata.json` — Model metadata.
- `artifacts/model_metrics.json` — Evaluation metrics.

## Limitations

Predictions are model-based estimates derived from historical I-94 observations. They are not guaranteed real-world traffic measurements or live traffic forecasts.
