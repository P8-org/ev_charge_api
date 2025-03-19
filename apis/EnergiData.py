import json
from pydantic import BaseModel, Json
import requests 
import datetime


class RequestDetail(BaseModel):
    startDate: datetime.date
    endDate: datetime.date
    dataset: str
    optional: str = ""
    filter_json: str = ""
    sort_data: str = ""
    offset: int = 0
    limit: int = 100


class EnergiDataInstance(BaseModel):
    HourUTC: str = ""
    HourDK: str = ""
    PriceArea: str = ""
    SpotPriceDKK: float = 0
    SpotPriceEUR: float = 0

class EnergiData:
    def __init__(self):
        self.data = []

    data: list[EnergiDataInstance]

    def call_api(self, rd: RequestDetail):
        base_url = "https://api.energidataservice.dk/dataset/"
        request_string = f'{rd.dataset}?start={rd.startDate}&end={rd.endDate}'

        if not rd.optional == "":
            request_string += f'&columns={rd.optional}'
        if not rd.filter_json == "":
            request_string += f'&filter={rd.filter_json}'
        if not rd.sort_data == "":
            request_string += f'&sort={rd.sort_data}'
        if not rd.offset == 0:
            request_string += f'&offset={rd.offset}'
        if not rd.limit == 100:
            request_string += f'&limit={rd.limit}'

        r = requests.get(base_url+request_string)
        r_json = r.json()

        for record in r_json["records"]:
            j = json.dumps(record)
            edi = EnergiDataInstance.model_validate_json(j)
            self.data.append(edi)
        
        












