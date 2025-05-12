from fastapi import APIRouter, Depends, HTTPException
import os
import requests
import zipfile
from io import BytesIO
from dotenv import load_dotenv
from RL.DQN import DQN_single, DQN_multi 
from apis.EnergiData import RequestDetail
import json

from sqlalchemy.orm import Session, joinedload
from models.models import UserEV
from database.db import get_db

load_dotenv()

router = APIRouter()

@router.get("/schedule/{ev_id}")
def run(ev_id: int, start_date, end_date, db: Session = Depends(get_db)): 
    ev: UserEV = db.query(UserEV).options(
        joinedload(UserEV.constraints),
        joinedload(UserEV.schedule),
        joinedload(UserEV.car_model)
    ).get(ev_id)

    if ev is None:
        raise HTTPException(status_code=404, detail=f"EV with id {ev_id} not found")


    rd = RequestDetail(
        startDate=start_date,
        endDate=end_date,
        dataset="Elspotprices",
        sort_data="HourDK ASC",
        filter_json=json.dumps({"PriceArea": ["DK1"]}),
    )


    constraint = ev.get_next_constraint()
    if not constraint:
        raise HTTPException(status_code=400, detail="No upcoming constraints")
    
    car = {
        'id': ev_id, 
        'charge_percentage': min(ev.current_charge / ev.car_model.battery_capacity * 100, 100),
        'min_percentage': constraint.target_percentage * 100,
        'charge': ev.current_charge,
        'max_charge_kw': ev.car_model.battery_capacity,
        'charge_speed': ev.max_charging_power,
        'constraints': {"start": constraint.start_time.hour, "end": constraint.end_time.hour} #how to do from ev????? constraint is the index in period; if input is date, find the index 
        # 'constraints': {"start": 15, "end": 17}
    }

    dqn_result = DQN_single.run_dqn(car=car,rd=rd)
    # dqn_result = DQN_multi.run_dqn(car=car,rd=rd) # need a different way to call
    if dqn_result == "No model trained":
        dqn_download_artifact()
        dqn_result = DQN_single.run_dqn(car=car,rd=rd)

    return dqn_result


@router.get("/train_status")
def dqn_train_status():
    r = requests.get("https://api.github.com/repos/P8-org/ev_charge_api/actions/workflows/ailab.yml/runs")
    status = r.json()["workflow_runs"][0]["status"]
    print(f"DQN Training Status: {status}")
    return status



@router.get("/download_dqn")
def dqn_download_artifact():

    OWNER = "P8-org"
    REPO = "ev_charge_api"
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    OUTPUT_DIR = "."

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
