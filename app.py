from contextlib import asynccontextmanager
from fastapi import FastAPI
from database.base import Base
from database.db import engine
from routers import users, evs
import datetime
import json

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

