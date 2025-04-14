from contextlib import asynccontextmanager
from fastapi import FastAPI
import requests
from rich import print
from database.base import Base
from database.db import engine
from routers import users, evs
import datetime
import json
import os

from apis.EnergiData import EnergiData, RequestDetail

@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs before startup of server
    Base.metadata.create_all(bind=engine) # create db
    yield
    # runs after shutdown oftserver
    print("app shutdown complete")


app = FastAPI(lifespan=lifespan)


app.include_router(users.router)
app.include_router(evs.router)

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
    EVENT_TYPE = "custom-event"

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    data = {
        "event_type": EVENT_TYPE,
        "client_payload": {
            "from": "fastapi",
            "info": "anything you want here"
        }
    }
    url = "https://api.github.com/repos/P8-org/ev_charge_api"
    r = requests.post(url, headers=headers, json=data)

@app.get("/dqn_train_status")
def dqn_train_status():
    r = requests.get("https://api.github.com/repos/P8-org/ev_charge_api/actions/workflows/ailab.yml/runs")
    # print(r.json()["workflow_runs"][0]["status"])
    status = r.json()["workflow_runs"][0]["status"]
    # return f"DQN Training Status: {status}"
    return status


