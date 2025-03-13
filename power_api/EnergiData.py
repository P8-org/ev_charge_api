from typing import Any, Optional
from pydantic import BaseModel, Json
import requests 
from power_api.BaseApi import BaseApiCall, RequestDetail
import datetime

# other name?
class RequestEnergiData(RequestDetail):
    dataset: str
    optional: Optional[str] = ""
    filter_json: Optional[str] = ""
    sort_data: Optional[str] = ""
    offset: Optional[int] = 100
    limit: Optional[int] = 100
    

class EnergiDataInstance(BaseModel):
    hourUTC: str
    hourDK: str
    priceArea: str = "DK1"
    version: str
    fuelAllocationMethod: str
    reportGrpCode: str
    productionType: str
    deliveryType: str
    production_MWh: Optional[float] = 0
    shareTotal: Optional[float] = 0
    shareGrid: Optional[float] = 0
    fuelConsumptionGJ: Optional[float] = 0
    CO2PerKWh: Optional[float] = 0
    CO2OriginPerKWh: Optional[float] = 0
    SO2PerKWh: Optional[float] = 0
    NOxPerKWh: Optional[float] = 0
    NMvocPerKWh: Optional[float] = 0
    CH4PerKWh: Optional[float] = 0
    COPerKWh: Optional[float] = 0
    N2OPerKWh: Optional[float] = 0
    slagPerKWh: Optional[float] = 0
    flyAshPerKWh: Optional[float] = 0
    particlesPerKWh: Optional[float] = 0
    wastePerKWh: Optional[float] = 0
    desulpPerKWh: Optional[float] = 0

class EnergiData(BaseApiCall):
    # data: dict[str, EnergiDataInstance] # key = hourUTC

    def call_api(self, rd: RequestDetail):
        if not isinstance(rd, RequestEnergiData):
            raise TypeError(f"Expected an instance of Parent or its subclass, got {type(rd).__name__}")
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
        return r.json()
        












