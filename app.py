from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from rich import print
from database.base import Base
from database.db import engine, get_db, seed_db
from routers import carmodels, evs, schedules, constraints, DQN
import datetime
import json
from modules.rl_short_term_scheduling import generate_schedule
from modules.linear_optimization_controller import adjust_rl_schedule
from modules.benchmark_prices import Benchmark
import numpy as np
import uvicorn

from apis.EnergiData import EnergiData, RequestDetail
from scheduler import scheduler
from websocket_manager import WSManager



@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs before startup of server
    Base.metadata.create_all(bind=engine) # create db
    seed_db()
    yield
    # runs after shutdown oftserver
    scheduler.shutdown()
    print("app shutdown complete")


app = FastAPI(lifespan=lifespan)


app.include_router(evs.router)
app.include_router(schedules.router)
app.include_router(constraints.router)
app.include_router(carmodels.router)
app.include_router(DQN.router, prefix="/dqn")

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
    lim = 0
    option = "HourUTC,PriceArea"
    fil = json.dumps({"PriceArea": ["DK1"]})
    sort = "HourUTC ASC"
    offset = 0
    formatted_time = (datetime.datetime.now() - datetime.timedelta(minutes=59)).strftime("%Y-%m-%dT%H:%M")

    # rd = RequestDetail(startDate=last_week, endDate=now, dataset="Elspotprices", optional=option, limit=lim, filter_json=fil, sort_data=sort, offset=offset)
    # rd = RequestDetail(startDate=last_week, endDate=now,dataset="Elspotprices",limit=lim, filter_json=fil)
    # rd = RequestDetail(startDate=last_week, endDate=now,dataset="Elspotprices", optional=option, filter_json=fil)
    rd = RequestDetail(startDate=formatted_time, dataset="Elspotprices", filter_json=fil, sort_data=sort)
    # rd = RequestDetail(startDate=last_week, dataset="Elspotprices", filter_json=fil, limit=lim)
    # e.call_api(rd)
    # return e.data
    return e.call_api(rd)



 

@app.get("/rl_schedule") # should also be replaced. also proof of concept :)
def schedule(num_hours: int, battery_level: float, battery_capacity: float, max_chargin_rate: float):
    formatted_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M")
    e = EnergiData()
    rd = RequestDetail(startDate=formatted_time, dataset="Elspotprices", filter_json=json.dumps({"PriceArea": ["DK1"]}), sort_data="HourDK ASC")
    response = e.call_api(rd)
    
    hour_dk = [record.HourDK for record in response]
    prices = [record.SpotPriceDKK / 1000 for record in response]
    schedule = generate_schedule(num_hours, battery_level, battery_capacity, max_chargin_rate, prices)
    adjusted_schedule = adjust_rl_schedule(schedule,battery_capacity - battery_level, max_chargin_rate)
    print(np.array(schedule))
    print(adjusted_schedule)
    b = Benchmark(adjusted_schedule,prices, battery_capacity - battery_level,max_chargin_rate)
    b.compare()
    return [{"time": h, "price": p, "charging": b} for h, p, b in zip(hour_dk, prices, adjusted_schedule)]

ws_manager = WSManager()

@app.middleware("http")
async def notify_websockets_on_change(request: Request, call_next):
    response = await call_next(request)

    if request.method in ["POST", "PUT", "PATCH", "DELETE"]:
        await ws_manager.broadcast(f"Change detected")

    return response

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    finally:
        await ws_manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app="app:app", reload=True, host="0.0.0.0")
