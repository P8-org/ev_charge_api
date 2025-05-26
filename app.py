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
def power():
    e = EnergiData()
    fil = json.dumps({"PriceArea": ["DK1"]})
    sort = "HourUTC ASC"

    formatted_time = (datetime.datetime.now() - datetime.timedelta(minutes=59)).strftime("%Y-%m-%dT%H:%M")
    rd = RequestDetail(startDate=formatted_time, dataset="Elspotprices", filter_json=fil, sort_data=sort)
    return e.call_api(rd)


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
