from fastapi import APIRouter, Depends
import os
import requests
import zipfile
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

@router.post("/train")
def dqn_start_train():
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN environment variable is not set")

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    data = {
        "ref": "main",  #change to current branch
        "inputs": {
            "from": "fastapi",
            "info": "Start Model Training"
        }
    }

    url = "https://api.github.com/repos/P8-org/ev_charge_api/actions/workflows/ailab.yml/dispatches"
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 204:
        print("✅ Workflow triggered successfully.")
    else:
        print(f"❌ Failed to trigger workflow: {response.status_code}")
        print(response.text)


@router.get("/dqn_train_status")
def dqn_train_status():
    r = requests.get("https://api.github.com/repos/P8-org/ev_charge_api/actions/workflows/ailab.yml/runs")
    status = r.json()["workflow_runs"][0]["status"]
    print(f"DQN Training Status: {status}")
    return status



@router.get("/download")
def dqn_download_artifact():

    OWNER = "P8-org"
    REPO = "ev_charge_api"
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    OUTPUT_DIR = "models"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    # Step 1: Get the latest workflow run
    base_url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions"
    workflow_runs_url = f"{base_url}/workflows/ailab.yml/runs"
    runs_response = requests.get(workflow_runs_url, headers=headers)
    runs_response.raise_for_status()
    latest_run = runs_response.json()["workflow_runs"][0]
    run_id = latest_run["id"]
    # print(run_id)

    # Step 2: Get artifacts for that run
    artifacts_url = f"{base_url}/runs/{run_id}/artifacts"
    artifacts_response = requests.get(artifacts_url, headers=headers)
    artifacts_response.raise_for_status()
    artifacts = artifacts_response.json()["artifacts"]
    print(artifacts)

    if not artifacts:
        raise Exception("No artifacts found for the latest workflow run.")

    # Step 3: Download the first artifact
    artifact = artifacts[0]
    artifact_id = artifact["id"]
    artifact_name = artifact["name"]
    download_url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/artifacts/{artifact_id}/zip"

    print(f"Downloading artifact: {artifact['name']}...")

    response = requests.get(download_url, headers=headers)
    response.raise_for_status()

# Save it locally
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    artifact_path = os.path.join(OUTPUT_DIR, f"{artifact['name']}.zip")
    with open(artifact_path, "wb") as f:
        f.write(response.content)

    print(f"Artifact downloaded to: {artifact_path}")

    # Step 4: Unzip the artifact
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(OUTPUT_DIR)

    print(f"Artifact unzipped to: {os.path.join(OUTPUT_DIR, artifact_name)}")
    os.remove(os.path.join(OUTPUT_DIR, f"{artifact['name']}.zip"))


    return"Artifact Updated"
