
# Pushes all deployment files to Hugging Face Space
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="tourism_project/deployment",      # local folder with Dockerfile, app.py, requirements.txt
    repo_id="carnage-colossus/Advanced-MLops",  # your HF Space
    repo_type="space",
    path_in_repo="",
)
print("Deployment files pushed to Hugging Face Space.")
