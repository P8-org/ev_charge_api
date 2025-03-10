from typing import Union
from fastapi import FastAPI
import requests

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/power")
def power():
    return requests.get('https://api.energidataservice.dk/dataset/DeclarationProduction?start=2022-05-01&end=2022-06-01&filter={"PriceArea":["DK1"]}').json()

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}




