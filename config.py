from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_DATASET = PROJECT_ROOT / "Metro_Interstate_Traffic_Volume.csv"

MODEL_PATH = MODEL_DIR / "traffic_pipeline_holiday_corrected.joblib"
METRICS_PATH = ARTIFACT_DIR / "holiday_corrected_app_metrics.json"
METADATA_PATH = ARTIFACT_DIR / "holiday_corrected_app_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "traffic_volume"

for directory in (DATA_DIR, MODEL_DIR, ARTIFACT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
