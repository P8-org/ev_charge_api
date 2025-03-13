import datetime
import json
from typing import Union
from fastapi import FastAPI
import requests

from power_api.EnergiData import EnergiData, RequestDetail

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

# @app.get("/power")
# def power():
#     return requests.get('https://api.energidataservice.dk/dataset/DeclarationProduction?start=2022-05-01&end=2022-06-01&filter={"PriceArea":["DK1"]}').json()

@app.get("/power")
def power():

    e = EnergiData()
    now = datetime.date.today() - datetime.timedelta(days=300)
    last_week = datetime.date.today() - datetime.timedelta(days=301)
    lim = 2
    option = "HourUTC,Production_MWh"
    fil = json.dumps({"PriceArea": ["DK1"]})
    sort = "HourUTC"
    offset = 0

    # rd = RequestDetail(startDate=last_week, endDate=now,dataset="DeclarationProduction", optional=option, limit=lim, filter_json=fil, sort_data=sort, offset=offset)
    rd = RequestDetail(startDate=last_week, endDate=now,dataset="Elspotprices",limit=lim, filter_json=fil)
    e.call_api(rd)
    return e.data

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}




