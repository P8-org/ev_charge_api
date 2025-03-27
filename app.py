from contextlib import asynccontextmanager
from fastapi import FastAPI
from database.base import Base
from database.db import engine
from routers import users, evs
import datetime
import json
from fastapi import FastAPI
import rl_scheduling

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
    lim = 24
    option = "HourUTC,PriceArea"
    fil = json.dumps({"PriceArea": ["DK1"]})
    sort = "HourUTC"
    offset = 0

    # rd = RequestDetail(startDate=last_week, endDate=now, dataset="Elspotprices", optional=option, limit=lim, filter_json=fil, sort_data=sort, offset=offset)
    # rd = RequestDetail(startDate=last_week, endDate=now,dataset="Elspotprices",limit=lim, filter_json=fil)
    # rd = RequestDetail(startDate=last_week, endDate=now,dataset="Elspotprices", optional=option, filter_json=fil)
    rd = RequestDetail(startDate=last_week, endDate=now,dataset="Elspotprices", filter_json=fil)
    # e.call_api(rd)
    # return e.data
    return e.get_today(rd)

@app.get("/rl_schedule") # should also be replaces. also proof of concept :)
def schedule(num_hours: int, battery_level: float, battery_capacity: float, max_chargin_rate: float):
    formatted_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M")
    e = EnergiData()
    rd = RequestDetail(startDate=formatted_time, dataset="Elspotprices", filter_json=json.dumps({"PriceArea": ["DK1"]}), sort_data="HourDK ASC")
    response = e.call_api(rd)
    
    hour_dk = [record.HourDK for record in response]
    prices = [record.SpotPriceDKK / 1000 for record in response]
    if (num_hours >= len(prices)): num_hours = len(prices)
    schedule_bool = rl_scheduling.get_schedule(num_hours, 0.2, 0.1, 100000, battery_level, battery_capacity, max_chargin_rate, prices, False)

    return [{"time": h, "price": p, "charging": b} for h, p, b in zip(hour_dk, prices, schedule_bool)]