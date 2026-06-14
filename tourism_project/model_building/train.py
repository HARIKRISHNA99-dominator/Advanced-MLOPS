# Production training script — runs headless in GitHub Actions CI/CD
import pandas as pd
import mlflow
import joblib
import os
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from huggingface_hub import HfApi, create_repo, login
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow connects to the server started by the GitHub Actions workflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_CICD_experiment")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)

DATASET_REPO   = "carnage-colossus/tourism-dataset"
MODEL_REPO     = "carnage-colossus/tourism-prediction-model"
MODEL_FILENAME = "best_tourism_prediction_model_v1.joblib"

# Load prepared train/test splits from Hugging Face
X_train = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
X_test  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
y_train = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").squeeze()
y_test  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").squeeze()
print(f"Data loaded — Train: {X_train.shape}, Test: {X_test.shape}")

# Scale all numeric features
numeric_features = X_train.columns.tolist()
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    remainder="passthrough"
)

# XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "xgbclassifier__n_estimators":  [100, 200],
    "xgbclassifier__learning_rate": [0.1, 0.2],
    "xgbclassifier__max_depth":     [3, 5],
}

pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, n_jobs=-1, scoring="roc_auc")
    grid_search.fit(X_train, y_train)

    # Log every tuned parameter combination as a nested run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_roc_auc", results["mean_test_score"][i])

    best_model = grid_search.best_estimator_
    y_pred     = best_model.predict(X_test)
    y_prob     = best_model.predict_proba(X_test)[:, 1]

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics({
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1":       f1_score(y_test, y_pred),
        "test_roc_auc":  roc_auc_score(y_test, y_prob),
    })

    # Save and log model artifact
    joblib.dump(best_model, MODEL_FILENAME)
    mlflow.log_artifact(MODEL_FILENAME)
    print("Model trained and logged.")

# Create model repo on HF if needed, then upload
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=MODEL_FILENAME,
    path_in_repo=MODEL_FILENAME,
    repo_id=MODEL_REPO,
    repo_type="model",
)
print(f"Model uploaded to {MODEL_REPO}")
