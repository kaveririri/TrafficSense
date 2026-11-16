from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_DATASET = DATA_DIR / "Metro_Interstate_Traffic_Volume.csv"

MODEL_PATH = MODEL_DIR / "traffic_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "traffic_volume"