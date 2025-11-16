import pandas as pd
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project folder
INPUT_CSV = PROJECT_ROOT / "data" / "youtube_ad_revenue_dataset.csv"
CLEAN_CSV = PROJECT_ROOT / "data" / "cleaned_data.csv"
FEATURES_JSON = PROJECT_ROOT / "models" / "features.json"

def load_data(path):
    df = pd.read_csv(path)
    return df

def handle_missing_values(df):
    numeric_cols = ['likes','comments','watch_time_minutes']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df

def remove_duplicates(df):
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    final_rows = df.shape[0]
    print(f"Duplicates removed :{initial_rows-final_rows}")
    return df

# ENCODE CATEGORICAL COLUMNS
def encode_categoricals(df):
    cat_cols = [c for c in ['category','device','country'] if c in df.columns]
    # keep all dummies (drop_first=False) to avoid accidental information loss
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df

def feature_engineering(df):
    # ensure date is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # small epsilon to avoid division by zero
    eps = 1e-6

    # engagement rate (likes+comments)/views
    df['engagement_rate'] = (df.get('likes', 0).fillna(0) + df.get('comments', 0).fillna(0)) / (df['views'].replace(0, eps))

    # watch ratio: how much of video was watched on average
    df['watch_ratio'] = df['watch_time_minutes'] / (df['video_length_minutes'] + eps)
    # clip to reasonable range to avoid huge outliers
    df['watch_ratio'] = df['watch_ratio'].clip(lower=0.0, upper=10.0)

    # date parts (fill NaT with 0 so training won't break)
    df['month'] = df['date'].dt.month.fillna(0).astype(int)
    df['day_of_week'] = df['date'].dt.dayofweek.fillna(0).astype(int)
    df['hour'] = df['date'].dt.hour.fillna(0).astype(int)

    return df

if __name__ == "__main__":
    df = load_data(INPUT_CSV)

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = feature_engineering(df)
    df = encode_categoricals(df)   # do encoding after feature engineering (keeps workflow consistent)

    # Keep 'ad_revenue_usd' (target) intact. Optionally drop columns not needed for training:
    # drop_cols = ['video_id', 'date']  # if you don't want these as features
    # df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Save cleaned CSV (project-relative path)
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"Cleaned dataset saved to {CLEAN_CSV}")

    # Save feature list (all columns except target) so other scripts know the exact expected columns
    features = [c for c in df.columns if c != 'ad_revenue_usd']
    FEATURES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURES_JSON, "w") as f:
        json.dump(features, f)
    print(f"Saved feature list ({len(features)} cols) to {FEATURES_JSON}")
