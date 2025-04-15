from contextlib import asynccontextmanager
from fastapi import FastAPI
import requests
from rich import print
from database.base import Base
from database.db import engine, seed_db
from routers import carmodels, evs, schedules, constraints
import datetime
import json
import os
import zipfile
from io import BytesIO
from dotenv import load_dotenv
from modules.rl_short_term_scheduling import generate_schedule
from modules.linear_optimization_controller import adjust_rl_schedule
from modules.benchmark_prices import Benchmark
import numpy as np

from apis.EnergiData import EnergiData, RequestDetail

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs before startup of server
    Base.metadata.create_all(bind=engine) # create db
    seed_db()
    yield
    # runs after shutdown oftserver
    print("app shutdown complete")


app = FastAPI(lifespan=lifespan)


app.include_router(evs.router)
app.include_router(schedules.router)
app.include_router(constraints.router)
app.include_router(carmodels.router)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/power")
def power(): # should be replaced, but proof-of-concept

    e = EnergiData()
    now = datetime.date(year=2024,month=4,day=18)
    last_week = datetime.date(year=2024,month=4,day=17)
    # last_week = "StartOfYear-P2Y"
    # now = "now"
    lim = 24
    option = "HourUTC,PriceArea"
    fil = json.dumps({"PriceArea": ["DK1"]})
    sort = "HourUTC"
    offset = 0

    # rd = RequestDetail(startDate=last_week, endDate=now, dataset="Elspotprices", optional=option, limit=lim, filter_json=fil, sort_data=sort, offset=offset)
    # rd = RequestDetail(startDate=last_week, endDate=now,dataset="Elspotprices",limit=lim, filter_json=fil)
    # rd = RequestDetail(startDate=last_week, endDate=now,dataset="Elspotprices", optional=option, filter_json=fil)
    rd = RequestDetail(startDate=last_week, endDate=now,dataset="Elspotprices", filter_json=fil, limit=lim)
    # rd = RequestDetail(startDate=last_week, dataset="Elspotprices", filter_json=fil, limit=lim)
    # e.call_api(rd)
    # return e.data
    return e.call_api(rd)


@app.get("/start_dqn_train")
def dqn_start_train():
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN environment variable is not set")

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    data = {
        "ref": "RL",  
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

@app.get("/dqn_train_status")
def dqn_train_status():
    r = requests.get("https://api.github.com/repos/P8-org/ev_charge_api/actions/workflows/ailab.yml/runs")
    status = r.json()["workflow_runs"][0]["status"]
    print(f"DQN Training Status: {status}")
    return status


@app.get("/dqn_download")
def dqn_download():

    OWNER = "P8-org"
    REPO = "ev_charge_api"
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    OUTPUT_DIR = "models"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    # Step 1: Get the latest workflow run
    workflow_runs_url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs"
    runs_response = requests.get(workflow_runs_url, headers=headers)
    runs_response.raise_for_status()
    latest_run = runs_response.json()["workflow_runs"][0]
    run_id = latest_run["id"]

    # Step 2: Get artifacts for that run
    artifacts_url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs/{run_id}/artifacts"
    artifacts_response = requests.get(artifacts_url, headers=headers)
    artifacts_response.raise_for_status()
    artifacts = artifacts_response.json()["artifacts"]

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


    return "Artifact Updated"

 
@app.get("/rl_schedule") # should also be replaced. also proof of concept :)
def schedule(num_hours: int, battery_level: float, battery_capacity: float, max_chargin_rate: float):
    formatted_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M")
    e = EnergiData()
    rd = RequestDetail(startDate=formatted_time, dataset="Elspotprices", filter_json=json.dumps({"PriceArea": ["DK1"]}), sort_data="HourDK ASC")
    response = e.call_api(rd)
    
    hour_dk = [record.HourDK for record in response]
    prices = [record.SpotPriceDKK / 1000 for record in response]
    schedule = generate_schedule(num_hours, battery_level, battery_capacity, max_chargin_rate, prices)
    adjusted_schedule = adjust_rl_schedule(schedule,battery_capacity, max_chargin_rate)
    print(np.array(schedule))
    print(adjusted_schedule)
    b = Benchmark(adjusted_schedule,prices,battery_capacity,max_chargin_rate)
    b.compare()
    return [{"time": h, "price": p, "charging": b} for h, p, b in zip(hour_dk, prices, adjusted_schedule)]
