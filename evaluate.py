# src/evaluate.py
import pandas as pd
import pickle
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import numpy as np

MODEL_PATH = Path("model") / "best_model.pkl"
DATA_PATH = Path("data") / "cleaned_data.csv"

def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {path.resolve()}")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path.resolve()}")
    return pd.read_csv(path)

def prepare_data_for_predict(df: pd.DataFrame, model):
    # drop target if present
    if "ad_revenue_usd" in df.columns:
        df = df.drop("ad_revenue_usd", axis=1)

    # If model exposes feature_names_in_, use it to pick columns in the same order
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise ValueError(
                "Missing columns required by the model:\n  - " + "\n  - ".join(missing) +
                "\n\nThis usually happens when the data was encoded differently than during training."
            )
        X = df[list(feature_names)].copy()
    else:
        # fallback: use numeric columns (best-effort)
        X = df.select_dtypes(include=[np.number, "bool"]).copy()
        if X.shape[1] == 0:
            raise ValueError("Could not infer feature columns for prediction. Model has no feature_names_in_.")
    # final NaN-fill (safe fallback)
    X = X.fillna(0)
    return X

def evaluate(model_path: str = None, data_path: str = None, save_predictions: bool = False, out_path: str = "predictions_test.csv"):
    model_path = Path(model_path) if model_path else MODEL_PATH
    data_path = Path(data_path) if data_path else DATA_PATH

    model = load_model(model_path)
    df = load_data(data_path)

    # Keep true target if present
    y = None
    if "ad_revenue_usd" in df.columns:
        y = df["ad_revenue_usd"].reset_index(drop=True)

    X = prepare_data_for_predict(df, model)

    preds = model.predict(X)
    preds = np.asarray(preds).ravel()

    if y is None:
        print("No true target found in data. Predictions computed but no metrics available.")
    else:
        if len(preds) != len(y):
            raise RuntimeError(f"Prediction length ({len(preds)}) does not match target length ({len(y)}).")
        mse = mean_squared_error(y, preds)
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        print("Evaluation on test set:")
        print(f" MSE : {mse:.6f}")
        print(f" MAE : {mae:.6f}")
        print(f" R2  : {r2:.6f}")

        # show first 10 aligned rows
        sample = pd.DataFrame({"y_true": y.values[:10], "y_pred": preds[:10]})
        print("\nSample predictions (first 10):")
        print(sample.to_string(index=False))

    if save_predictions:
        out_df = pd.DataFrame({"pred": preds})
        if y is not None:
            out_df.insert(0, "true", y.values)
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved predictions to: {Path(out_path).resolve()}")

if __name__ == "__main__":
    # CLI usage:
    # python src/evaluate.py [model_path] [data_path] [save_predictions(true/false)] [out_path]
    args = sys.argv[1:]
    model_p = args[0] if len(args) >= 1 else None
    data_p = args[1] if len(args) >= 2 else None
    save_flag = (args[2].lower() == "true") if len(args) >= 3 else False
    out_p = args[3] if len(args) >= 4 else "predictions_test.csv"

    evaluate(model_path=model_p, data_path=data_p, save_predictions=save_flag, out_path=out_p)
