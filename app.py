from contextlib import asynccontextmanager
from fastapi import FastAPI
import requests
from database.base import Base
from database.db import engine
from routers import users, evs

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
def power():
    return requests.get('https://api.energidataservice.dk/dataset/DeclarationProduction?start=2022-05-01&end=2022-06-01&filter={"PriceArea":["DK1"]}').json()

