import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------
# LOAD CLEANED DATA
# ---------------------------------------------------------
df = pd.read_csv(
    r"C:\Users\Devansh Gera\Desktop\content_monetization_modeler\data\cleaned_data.csv"
)

# ---------------------------------------------------------
# SPLIT FEATURES & TARGET
# ---------------------------------------------------------
y = df["ad_revenue_usd"]
X = df.drop("ad_revenue_usd", axis=1)

# Remove non-numeric useless columns (ID, Date)
X = X.drop(columns=["video_id", "date"], errors="ignore")

# Keep only numeric features
X = X.select_dtypes(include=[np.number])

# Fill any tiny leftovers
X = X.fillna(0)

# ---------------------------------------------------------
# TRAIN–TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------------

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_pred = lr_pred.clip(min=0)
lr_r2 = r2_score(y_test, lr_pred)

# Lasso Regression
lasso_model = Lasso(alpha=0.1, max_iter=10000, random_state=42)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_pred = lasso_pred.clip(0)
lasso_r2 = r2_score(y_test, lasso_pred)

# Ridge Regression
ridge_model = Ridge(random_state=42)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_pred = ridge_pred.clip(min=0)
ridge_r2 = r2_score(y_test, ridge_pred)

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_pred = dt_pred.clip(min=0)
dt_r2 = r2_score(y_test, dt_pred)

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_pred = rf_pred.clip(min=0)
rf_r2 = r2_score(y_test, rf_pred)

# ---------------------------------------------------------
# SELECT BEST MODEL
# ---------------------------------------------------------
scores = {
    "linear": lr_r2,
    "lasso": lasso_r2,
    "ridge": ridge_r2,
    "decision_tree": dt_r2,
    "random_forest": rf_r2
}

models = {
    "linear": lr_model,
    "lasso": lasso_model,
    "ridge": ridge_model,
    "decision_tree": dt_model,
    "random_forest": rf_model
}

best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]

# ---------------------------------------------------------
# SAVE BEST MODEL
# ---------------------------------------------------------
save_path = r"C:\Users\Devansh Gera\Desktop\content_monetization_modeler\model\best_model.pkl"

with open(save_path, "wb") as f:
    pickle.dump(best_model, f)

print(f"Best model selected: {best_model_name} (R2={scores[best_model_name]:.4f})")
print(f"Model saved at: {save_path}")
