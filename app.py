import datetime
import json
from typing import Union
from fastapi import FastAPI

from power_api.EnergiData import EnergiData, RequestDetail
from test.test_power_api import test_api_call

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/power")
def power():

    e = EnergiData()
    now = datetime.date(year=2024,month=4,day=18)
    last_week = datetime.date(year=2024,month=4,day=17)
    print(now)
    lim = 1
    option = "PriceArea"
    fil = json.dumps({"PriceArea": ["DK1"]})
    sort = "HourUTC"
    offset = 0

    # rd = RequestDetail(startDate=last_week, endDate=now,dataset="DeclarationProduction", optional=option, limit=lim, filter_json=fil, sort_data=sort, offset=offset)
    # rd = RequestDetail(startDate=last_week, endDate=now,dataset="Elspotprices",limit=lim, filter_json=fil)
    rd = RequestDetail(startDate=last_week, endDate=now,dataset="Elspotprices", optional=option)
    e.call_api(rd)
    return e.data

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}



