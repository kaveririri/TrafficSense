"""Train and evaluate TrafficSense models from the bundled CSV."""
import json
from config import DEFAULT_DATASET
from src.data_loader import load_dataset
from src.model_training import train_and_evaluate

if __name__ == "__main__":
    raw = load_dataset(DEFAULT_DATASET)
    result = train_and_evaluate(raw)
    print("Best model:", result["best_model"])
    print(result["metrics"].to_string(index=False))
    print("Saved model:", result["model_path"])
    print("Metadata:", json.dumps(result["metadata"], indent=2))
