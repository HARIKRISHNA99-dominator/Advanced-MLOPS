import pandas as pd
import os
import joblib
import mlflow
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from huggingface_hub import login, HfApi

# --- MLflow and HF Setup ---
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_CICD_experiment")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)

# --- Load Data ---
X_train = pd.read_csv("tourism_project/data/X_train.csv")
X_test = pd.read_csv("tourism_project/data/X_test.csv")
y_train = pd.read_csv("tourism_project/data/y_train.csv").squeeze()
y_test = pd.read_csv("tourism_project/data/y_test.csv").squeeze()

# --- Categorical Encoding ---
# Dynamically encode any object/string columns found in the CSVs
cat_cols = X_train.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# --- Preprocessing ---
numeric_features = X_train.columns.tolist()
preprocessor = make_column_transformer((StandardScaler(), numeric_features), remainder="passthrough")

# --- Model Setup ---
xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=-1)
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__learning_rate": [0.1, 0.2],
    "xgbclassifier__max_depth": [3, 5]
}
pipeline = make_pipeline(preprocessor, xgb_model)

# --- Training ---
with mlflow.start_run():
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring="roc_auc")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    mlflow.log_metrics({
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_prob)
    })
    
    model_file = "best_tourism_prediction_model_v1.joblib"
    joblib.dump(best_model, model_file)
    mlflow.log_artifact(model_file)

# --- Upload ---
repo_id = "carnage-colossus/tourism-prediction-model"
try:
    api.upload_file(path_or_fileobj=model_file, path_in_repo=model_file, repo_id=repo_id, repo_type="model")
    print(f"Success! Model uploaded to {repo_id}")
except Exception as e:
    print(f"Training finished, but Hugging Face upload failed: {e}")
