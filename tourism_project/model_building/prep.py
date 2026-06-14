# Loads data from HF, cleans, encodes, splits, and re-uploads train/test sets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_REPO = "carnage-colossus/tourism-dataset"
DATASET_PATH = f"hf://datasets/{DATASET_REPO}/tourism.csv"

# Load raw dataset from Hugging Face
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {df.shape}")

# Drop identifier columns (not useful for modeling)
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True)

# Label-encode all categorical columns (alphabetical order = LabelEncoder default)
cat_cols = ["TypeofContact", "Occupation", "Gender",
            "ProductPitched", "MaritalStatus", "Designation"]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    print(f"Encoded: {col}")

# Separate features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Stratified 80/20 split to preserve class balance (19% positive class)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"y_train positive rate: {y_train.mean():.2%}")
print(f"y_test  positive rate: {y_test.mean():.2%}")

# Save splits locally
X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv",   index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv",   index=False)

# Upload all four splits back to the HF Dataset Hub
for fname in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=fname,
        path_in_repo=fname,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Uploaded: {fname}")

print("Data preparation complete.")
