# Registers the raw tourism.csv to the Hugging Face Dataset Hub
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

REPO_ID   = "carnage-colossus/tourism-package-data"
REPO_TYPE = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the dataset repo if it does not already exist
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Repo '{REPO_ID}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False)
    print(f"Repo '{REPO_ID}' created.")

# Upload the entire data folder to Hugging Face
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)
print("Dataset uploaded to Hugging Face successfully.")
