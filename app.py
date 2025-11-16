#### GIVES NEGATIVE VALUE

# # app.py
# # Streamlit app to predict YouTube ad revenue (USD + INR) using saved model.
# # Place this file in your project root (same level as data/ and model/).
# # Run: streamlit run app.py

# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# from pathlib import Path

# MODEL_PATH = Path("model/best_model.pkl")
# CLEANED_DATA_PATH = Path("data/cleaned_data.csv")  # used only to infer categorical options if available

# st.set_page_config(page_title="YouTube Revenue Predictor", layout="centered")

# # --- helpers ----------------------------------------------------------------
# @st.cache_resource
# def load_model(path: Path):
#     with open(path, "rb") as f:
#         model = pickle.load(f)
#     return model

# def extract_cat_options_from_feature_names(feature_names, prefix):
#     """From list of feature names like 'category_Music' return ['Music', ...]."""
#     opts = []
#     for fn in feature_names:
#         if fn.startswith(prefix + "_"):
#             opts.append(fn[len(prefix) + 1 :])
#     return sorted(opts)

# def build_input_row(feature_names, numeric_inputs, cat_selections):
#     """Create a single-row DataFrame matching model.feature_names_in_ order."""
#     # default zero for all model features
#     row = {fn: 0 for fn in feature_names}

#     # fill numeric inputs if model expects them
#     for k, v in numeric_inputs.items():
#         if k in row:
#             row[k] = float(v)

#     # compute derived features (safe)
#     # engagement_rate = (likes + comments) / views
#     views = numeric_inputs.get("views", 0) or 0
#     likes = numeric_inputs.get("likes", 0) or 0
#     comments = numeric_inputs.get("comments", 0) or 0
#     watch_time_minutes = numeric_inputs.get("watch_time_minutes", 0) or 0
#     video_length_minutes = numeric_inputs.get("video_length_minutes", 0) or 0

#     eps = 1e-9
#     engagement_rate = (likes + comments) / (views + eps)
#     watch_ratio = watch_time_minutes / (video_length_minutes + eps)

#     if "engagement_rate" in row:
#         row["engagement_rate"] = float(engagement_rate)
#     if "watch_ratio" in row:
#         row["watch_ratio"] = float(watch_ratio)

#     # If model expects an estimated_revenue_per_view or revenue_per_view and it's missing,
#     # create a conservative estimate (small value) rather than leaking target.
#     # We compute a small estimate: (watch_time_minutes / (views + eps)) * 0.01
#     est_rev_per_view = (watch_time_minutes / (views + eps)) * 0.01
#     if "estimated_revenue_per_view" in row and row["estimated_revenue_per_view"] == 0:
#         row["estimated_revenue_per_view"] = float(est_rev_per_view)
#     if "revenue_per_view" in row and row["revenue_per_view"] == 0:
#         row["revenue_per_view"] = float(est_rev_per_view)

#     # set categorical dummies according to cat_selections (e.g. {'category': 'Music'}).
#     # model feature names expected like 'category_Music' or 'device_Mobile'
#     for cat_col, sel in cat_selections.items():
#         if sel is None:
#             continue
#         key = f"{cat_col}_{sel}"
#         if key in row:
#             row[key] = 1
#         else:
#             # if model used drop_first=True during training, the base category won't have a column.
#             # In that case nothing to set (all zeros represent base).
#             pass

#     # order columns same as feature_names
#     ordered = {fn: row.get(fn, 0) for fn in feature_names}
#     return pd.DataFrame([ordered])

# # --- load model & feature info ----------------------------------------------
# model = load_model(MODEL_PATH)
# feature_names = list(map(str, getattr(model, "feature_names_in_", [])))

# # extract categorical options from model feature names (preferred)
# category_options = extract_cat_options_from_feature_names(feature_names, "category")
# device_options = extract_cat_options_from_feature_names(feature_names, "device")
# country_options = extract_cat_options_from_feature_names(feature_names, "country")

# # fallback: if we couldn't infer options from model, try to read original cleaned_data.csv
# if not category_options or not device_options or not country_options:
#     try:
#         df_raw = pd.read_csv(CLEANED_DATA_PATH)
#     except Exception:
#         df_raw = None
#     if df_raw is not None:
#         if (not category_options) and ("category" in df_raw.columns):
#             category_options = sorted(df_raw["category"].dropna().unique().astype(str).tolist())
#         if (not device_options) and ("device" in df_raw.columns):
#             device_options = sorted(df_raw["device"].dropna().unique().astype(str).tolist())
#         if (not country_options) and ("country" in df_raw.columns):
#             country_options = sorted(df_raw["country"].dropna().unique().astype(str).tolist())

# # final default if still empty
# if not category_options:
#     category_options = ["Other"]
# if not device_options:
#     device_options = ["Mobile", "TV", "Tablet"]
# if not country_options:
#     country_options = ["US", "IN"]

# # --- UI ---------------------------------------------------------------------
# st.title("YouTube Ad Revenue Predictor")

# page = st.sidebar.radio("Go to", ["Overview", "Predict Revenue", "Insights"])

# # small sidebar control: INR conversion
# inr_rate = st.sidebar.number_input("USD → INR conversion rate", min_value=1.0, value=83.0, step=0.5, format="%.2f")

# if page == "Overview":
#     st.header("Overview")
#     st.write("Simple video-level revenue estimator. Choose Predict Revenue to test the model.")

# elif page == "Predict Revenue":
#     st.header("Predict Revenue")

#     # numeric inputs
#     views = st.number_input("Views", min_value=0, value=10000, step=1)
#     likes = st.number_input("Likes", min_value=0, value=500, step=1)
#     comments = st.number_input("Comments", min_value=0, value=50, step=1)
#     watch_time_minutes = st.number_input("Watch time (minutes)", min_value=0.0, value=200.0, step=0.1)
#     video_length_minutes = st.number_input("Video length (minutes)", min_value=0.1, value=10.0, step=0.1)
#     subscribers = st.number_input("Channel subscribers", min_value=0, value=10000, step=1)

#     # categorical inputs (options inferred)
#     category = st.selectbox("Category", category_options)
#     device = st.selectbox("Device", device_options)
#     country = st.selectbox("Country", country_options)

#     # time features
#     month = st.slider("Month (1-12)", 1, 12, 9)
#     day_of_week = st.slider("Day of week (0=Mon..6=Sun)", 0, 6, 1)
#     hour = st.slider("Upload hour (0-23)", 0, 23, 10)

#     st.markdown("---")
#     if st.button("Predict revenue"):
#         # prepare numeric & categorical maps
#         numeric_inputs = {
#             "views": views,
#             "likes": likes,
#             "comments": comments,
#             "watch_time_minutes": watch_time_minutes,
#             "video_length_minutes": video_length_minutes,
#             "subscribers": subscribers,
#             "month": month,
#             "day_of_week": day_of_week,
#             "hour": hour,
#         }
#         cat_selections = {"category": str(category), "device": str(device), "country": str(country)}

#         # build input row
#         if not feature_names:
#             st.error("Model does not expose feature names; cannot safely prepare input.")
#         else:
#             try:
#                 input_df = build_input_row(feature_names, numeric_inputs, cat_selections)
#                 # ensure numeric dtype
#                 input_df = input_df.astype(float)

#                 # predict
#                 pred_usd = float(model.predict(input_df)[0])

#                 # ensure non-negative and reasonable (clip small negatives to zero)
#                 if pred_usd < 0 and pred_usd > -1e-6:
#                     pred_usd = 0.0

#                 pred_inr = pred_usd * float(inr_rate)

#                 st.success("Prediction complete")
#                 col1, col2 = st.columns(2)
#                 col1.metric("Predicted ad revenue (USD)", f"${pred_usd:,.2f}")
#                 col2.metric("Predicted ad revenue (INR)", f"₹{pred_inr:,.2f}")

#                 st.markdown("#### Inputs that influence prediction:")
#                 st.write(f"- Views: {int(views):,}")
#                 st.write(f"- Engagement rate: {( (likes + comments) / (views + 1e-9) ):0.5f}")
#                 st.write(f"- Watch ratio: {(watch_time_minutes / (video_length_minutes + 1e-9)):0.6f}")
#                 # show used estimated revenue-per-view if present in features
#                 if "revenue_per_view" in feature_names or "estimated_revenue_per_view" in feature_names:
#                     # estimate displayed (same as used in prepare)
#                     est_rpv = (watch_time_minutes / (views + 1e-9)) * 0.01
#                     st.write(f"- Estimated revenue per view used: {est_rpv:0.5f}")

#             except Exception as e:
#                 st.error("Prediction failed. See debug info below.")
#                 st.exception(e)

# elif page == "Insights":
#     st.header("Insights")
#     # model name
#     st.write(f"Model: {type(model).__name__}")

#     # feature importance if available
#     if hasattr(model, "feature_importances_"):
#         fi = np.array(model.feature_importances_, dtype=float)
#         # pair with feature names if available
#         if feature_names and len(feature_names) == fi.shape[0]:
#             fi_series = pd.Series(fi, index=feature_names).sort_values(ascending=False)
#         else:
#             fi_series = pd.Series(fi).sort_values(ascending=False)

#         st.markdown("Top features (by importance):")
#         for name, val in fi_series.head(12).items():
#             st.write(f"- {name}: {val:0.4f}")
#     else:
#         st.write("Feature importances not available for this model.")

#     # list expected model features
#     if feature_names:
#         st.markdown("Model expects these features (in order):")
#         st.write(feature_names)

# # app.py
# # Streamlit app for YouTube ad revenue prediction (USD + INR)
# # Place this file at project root (same level as data/, model/, preprocess.py).
# # Run: streamlit run app.py

### GIVES ZERO VALUE FOR NEGATIVE

# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# from pathlib import Path

# # --- Config ---
# MODEL_PATH = Path("model") / "best_model.pkl"
# DATA_PATH = Path("data") / "cleaned_data.csv"  # used only to load lists of categories if needed
# DEFAULT_USD_TO_INR = 83.0
# EPS = 1e-9  # tiny epsilon

# st.set_page_config(page_title="YouTube Ad Revenue Predictor", layout="centered")

# # --- Utilities ---
# @st.cache_resource
# def load_model(path: Path):
#     with open(path, "rb") as f:
#         model = pickle.load(f)
#     return model

# def safe_numeric(x, default=0.0):
#     try:
#         return float(x)
#     except Exception:
#         return default

# def prepare_input(raw: dict, model_feature_names: np.ndarray, df_clean_path: Path = None):
#     """
#     raw: dict with keys for raw (unencoded) features including categorical keys:
#          'category', 'device', 'country' and numeric keys used in preprocess/feature_engineering
#     model_feature_names: array of feature names the model expects (order matters)
#     df_clean_path: optional path to cleaned_data.csv to extract categories if needed (not required)
#     Returns DataFrame with single row aligned to model_feature_names (missing columns filled with 0).
#     """
#     # start from raw dict -> DataFrame
#     df = pd.DataFrame([raw]).copy()

#     # Make sure numeric columns exist and are numeric
#     numeric_cols = [
#         "views", "likes", "comments", "watch_time_minutes", "video_length_minutes",
#         "subscribers", "month", "day_of_week", "hour",
#         "engagement_rate", "watch_ratio"
#     ]
#     for c in numeric_cols:
#         if c not in df.columns:
#             df[c] = 0.0
#         df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

#     # Ensure categorical columns exist
#     cat_cols = ["category", "device", "country"]
#     for c in cat_cols:
#         if c not in df.columns:
#             df[c] = "Unknown"

#     # One-hot encode the categorical variables — then align to model_feature_names
#     df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

#     # Reindex to model features: keep only columns present in model_feature_names, fill missing with 0
#     # Some training pipelines may have used drop_first=True; because we don't know, reindex handles missing columns
#     model_cols = list(model_feature_names)
#     aligned = pd.DataFrame(columns=model_cols)
#     # Fill existing intersections
#     for col in model_cols:
#         if col in df_encoded.columns:
#             aligned.loc[0, col] = df_encoded.loc[0, col]
#         else:
#             aligned.loc[0, col] = 0.0

#     # Ensure numeric dtype
#     aligned = aligned.astype(float).fillna(0.0)

#     return aligned

# def clamp_prediction(val):
#     # ensure non-negative; preserve small positives; round to 2 decimals
#     if np.isnan(val):
#         return 0.0
#     if val < 0:
#         return 0.0
#     return float(round(val, 2))

# # --- Load model (cached) ---
# if not MODEL_PATH.exists():
#     st.error(f"Model not found at {MODEL_PATH}. Train and save a model first.")
#     st.stop()

# model = load_model(MODEL_PATH)
# model_name = type(model).__name__
# feature_names_in = getattr(model, "feature_names_in_", None)

# # --- Sidebar: minimal controls (navigation + conversion rate) ---
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Overview", "Predict Revenue", "Insights"])

# st.sidebar.markdown("---")
# usd_to_inr = st.sidebar.number_input("USD → INR conversion rate", value=float(DEFAULT_USD_TO_INR), step=0.5, format="%.2f")

# # --- Pages ---
# if page == "Overview":
#     st.title("YouTube Ad Revenue Predictor")
#     st.write(
#         "Simple app to estimate ad revenue for a single video using the trained model."
#     )
#     st.write("Enter video metrics in Predict Revenue. Insights has model info.")
#     st.write("Outputs: predicted revenue in USD and INR (converted using rate in sidebar).")

# elif page == "Predict Revenue":
#     st.title("Predict Revenue")

#     # Input fields (mirror preprocessing features)
#     col1, col2 = st.columns(2)
#     with col1:
#         views = st.number_input("Views", min_value=0, value=1000, step=1)
#         likes = st.number_input("Likes", min_value=0, value=100, step=1)
#         comments = st.number_input("Comments", min_value=0, value=10, step=1)
#         watch_time_minutes = st.number_input("Watch time (minutes)", min_value=0.0, value=100.0, step=0.1, format="%.2f")
#         video_length_minutes = st.number_input("Video length (minutes)", min_value=0.1, value=5.0, step=0.1, format="%.2f")
#     with col2:
#         subscribers = st.number_input("Channel subscribers", min_value=0, value=10000, step=1)
#         month = st.slider("Month (1-12)", min_value=1, max_value=12, value=9)
#         day_of_week = st.slider("Day of week (0=Mon..6=Sun)", min_value=0, max_value=6, value=1)
#         hour = st.slider("Upload hour (0-23)", min_value=0, max_value=23, value=10)

#     # Categories: try to load candidate values from cleaned data if available; otherwise show small default list
#     default_categories = ["Education", "Entertainment", "Gaming", "Lifestyle", "Music", "Tech"]
#     default_devices = ["Mobile", "TV", "Tablet"]
#     default_countries = ["US", "IN", "CA", "UK", "DE"]

#     if DATA_PATH.exists():
#         try:
#             df_clean = pd.read_csv(DATA_PATH)
#             if "category" in df_clean.columns and df_clean["category"].nunique() > 0:
#                 category_choices = sorted(df_clean["category"].dropna().unique().tolist())
#             else:
#                 category_choices = default_categories

#             if "device" in df_clean.columns and df_clean["device"].nunique() > 0:
#                 device_choices = sorted(df_clean["device"].dropna().unique().tolist())
#             else:
#                 device_choices = default_devices

#             if "country" in df_clean.columns and df_clean["country"].nunique() > 0:
#                 country_choices = sorted(df_clean["country"].dropna().unique().tolist())
#             else:
#                 country_choices = default_countries
#         except Exception:
#             category_choices, device_choices, country_choices = default_categories, default_devices, default_countries
#     else:
#         category_choices, device_choices, country_choices = default_categories, default_devices, default_countries

#     category = st.selectbox("Category", category_choices)
#     device = st.selectbox("Device", device_choices)
#     country = st.selectbox("Country", country_choices)

#     # Derived features (same logic as preprocess)
#     eps = EPS
#     engagement_rate = (likes + comments) / (views + eps)
#     watch_ratio = watch_time_minutes / (video_length_minutes + eps)

#     st.markdown("---")
#     st.subheader("Input preview")
#     preview = {
#         "views": int(views),
#         "likes": int(likes),
#         "comments": int(comments),
#         "watch_time_minutes": float(watch_time_minutes),
#         "video_length_minutes": float(video_length_minutes),
#         "subscribers": int(subscribers),
#         "category": category,
#         "device": device,
#         "country": country,
#         "month": int(month),
#         "day_of_week": int(day_of_week),
#         "hour": int(hour),
#         "engagement_rate": float(engagement_rate),
#         "watch_ratio": float(watch_ratio),
#     }
#     st.dataframe(pd.DataFrame([preview]).T.rename(columns={0: "value"}))

#     if st.button("Predict revenue"):
#         # Prepare input aligned to model features
#         if feature_names_in is None:
#             st.error("Model does not expose feature names (feature_names_in_). Cannot safely align inputs.")
#         else:
#             try:
#                 X_for_model = prepare_input(preview, feature_names_in, df_clean_path=DATA_PATH)
#                 pred = model.predict(X_for_model)[0]
#                 pred_usd = clamp_prediction(pred)

#                 # If predicted extremely small (==0) and inputs indicate non-zero views, we still show 0.
#                 # But to avoid negative we clamped already.

#                 pred_inr = round(pred_usd * float(usd_to_inr), 2)

#                 st.success("Prediction complete")
#                 col_a, col_b = st.columns(2)
#                 with col_a:
#                     st.metric("Predicted ad revenue (USD)", f"${pred_usd:,.2f}")
#                 with col_b:
#                     st.metric("Predicted ad revenue (INR)", f"₹{pred_inr:,.2f}")

#                 # Show influence basics
#                 st.markdown("#### Inputs that influence prediction")
#                 st.write(f"- Views: {int(views):,}")
#                 st.write(f"- Engagement rate: {engagement_rate:.5f}")
#                 st.write(f"- Watch ratio: {watch_ratio:.5f}")
#                 # If the model exposes feature_importances_ we can show top contribution placeholders
#                 importances = getattr(model, "feature_importances_", None)
#                 if importances is not None and len(importances) == len(feature_names_in):
#                     fi = pd.Series(importances, index=feature_names_in).sort_values(ascending=False).head(6)
#                     st.write("- Estimated top features (by model):")
#                     for idx, val in fi.items():
#                         st.write(f"  - {idx}: {val:.4f}")
#                 else:
#                     st.write("- (Model feature importances not available)")

#                 # If prediction was clamped from negative, warn
#                 if pred < 0:
#                     st.warning("Model produced a negative value which was clamped to 0. Check training data or model.")
#             except Exception as e:
#                 st.exception(e)

# elif page == "Insights":
#     st.title("Insights")
#     st.write(f"Model: {model_name}")
#     # Show top features if available
#     importances = getattr(model, "feature_importances_", None)
#     if importances is not None and feature_names_in is not None:
#         fi = pd.Series(importances, index=feature_names_in).sort_values(ascending=False)
#         st.subheader("Top features (by importance)")
#         for name, val in fi.head(20).items():
#             st.write(f"- {name}: {val:.4f}")
#     else:
#         st.write("Feature importances not available for this model.")
#     st.markdown("---")
#     st.write("Notes:")
#     st.write("- App aligns one-hot encoded inputs to the model's expected feature names.")
#     st.write("- Negative predictions are clamped to 0 USD to avoid nonsensical output.")


###NEW CODE
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ------------------- CONFIG --------------------
st.set_page_config(page_title="YouTube Revenue Predictor", layout="centered")

# ------------------- TITLE ---------------------
st.title("🎬 YouTube Revenue Predictor")

# ------------------- MODEL LOADING ---------------------
@st.cache_resource
def load_model():
    with open("model/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Revenue", "Insights"])

# ------------------- HOME PAGE ---------------------
if page == "Home":
    st.header("Welcome")
    st.write("""
    This application predicts **YouTube ad revenue** based on video performance,  
    engagement, viewer device and country, watch time, and channel strength.
    """)

# ------------------- PREDICT PAGE ---------------------
elif page == "Predict Revenue":
    st.header("Predict revenue for a single video")

    # ------- INPUTS -------
    views = st.number_input("Views", min_value=0, value=5000)
    likes = st.number_input("Likes", min_value=0, value=200)
    comments = st.number_input("Comments", min_value=0, value=50)
    watch_time_minutes = st.number_input("Watch Time (minutes)", min_value=0.0, value=300.0)
    video_length_minutes = st.number_input("Video Length (minutes)", min_value=0.1, value=10.0)
    subscribers = st.number_input("Subscribers", min_value=0, value=10000)

    category = st.selectbox("Category", ["Education", "Entertainment", "Gaming", "Lifestyle", "Music", "Tech"])
    device = st.selectbox("Device", ["Mobile", "TV", "Tablet"])
    country = st.selectbox("Country", ["US", "IN", "CA", "UK", "DE"])

    month = st.slider("Month", 1, 12, 6)
    day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 2)
    hour = st.slider("Upload Hour", 0, 23, 12)

    # ------- FEATURE ENGINEERING RE-CREATION -------
    eps = 1e-9
    engagement_rate = (likes + comments) / (views + eps)
    watch_ratio = watch_time_minutes / (video_length_minutes + eps)

    # ------- BUILD INPUT DF -------
    input_dict = {
        "views": views,
        "likes": likes,
        "comments": comments,
        "watch_time_minutes": watch_time_minutes,
        "video_length_minutes": video_length_minutes,
        "subscribers": subscribers,
        "engagement_rate": engagement_rate,
        "watch_ratio": watch_ratio,
        "month": month,
        "day_of_week": day_of_week,
        "hour": hour,
    }

    # Create DataFrame
    input_df = pd.DataFrame([input_dict])

    # -------------- HANDLE ONE-HOT ENCODED COLUMNS --------------
    # Get all features the model was trained on
    model_features = list(model.feature_names_in_)

    # Create empty full-row with all columns model expects
    full_row = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)

    # Insert numeric features
    for col in input_df.columns:
        if col in full_row.columns:
            full_row[col] = input_df[col].values

    # Add categorical encodings: category_, device_, country_
    def add_cat(prefix, value):
        colname = f"{prefix}_{value}"
        if colname in full_row.columns:
            full_row[colname] = 1

    add_cat("category", category)
    add_cat("device", device)
    add_cat("country", country)

    st.write("Preview of final model-ready features:")
    st.dataframe(full_row)

    # -------------- PREDICT --------------
    if st.button("Predict Revenue"):
        pred = model.predict(full_row)[0]

        # avoid negative revenue
        pred = max(pred, 0)

        # convert to INR
        pred_inr = pred * 83  # approx conversion

        st.success(f"Estimated Revenue: **${pred:.2f} USD**")
        st.success(f"Estimated Revenue: **₹{pred_inr:.2f} INR**")

# ------------------- INSIGHTS PAGE ---------------------
elif page == "Insights":
    st.header("Model Insights")

    st.write("### 1. Model Used")
    st.write(f"- The prediction is generated using your **best-performing regression model**, trained on cleaned & engineered dataset.")

    st.write("### 2. Key Influencing Factors")
    st.write("""
    These features contribute most to predicting revenue:
    - Views  
    - Watch Time  
    - Engagement Rate  
    - Watch Ratio  
    - Subscriber Count  
    """)

    st.write("### 3. Why Revenue May Change")
    st.write("""
    Revenue depends on:
    - Viewer country  
    - Ad demand  
    - Watch duration  
    - Device (Mobile vs TV etc.)  
    """)

    st.write("These insights help explain why some videos earn more than others.")
