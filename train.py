from src.config import DATA_PATH, MODEL_PATH
from src.data import load_data, select_features
from src.model import train_model

if __name__ == "__main__":
    data = load_data(DATA_PATH)
    bundle = train_model(data, select_features(data), MODEL_PATH)
    print("Model saved to", MODEL_PATH)
    print("ROC-AUC:", round(bundle["metrics"]["roc_auc"], 4))
    print("PR-AUC:", round(bundle["metrics"]["pr_auc"], 4))

