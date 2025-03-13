import datetime
import json
from typing import Union
from fastapi import FastAPI
import requests

from power_api.EnergiData import EnergiData, RequestEnergiData

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
    lim = 7
    option = "HourUTC,Production_MWh"
    fil = json.dumps({"PriceArea": ["DK1"]})
    sort = "HourUTC"
    offset = 0

    rd = RequestEnergiData(startDate=last_week, endDate=now,dataset="DeclarationProduction", optional=option, limit=lim, filter_json=fil, sort_data=sort, offset=offset)

    return e.call_api(rd)

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}




