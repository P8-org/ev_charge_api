from typing import Optional
from pydantic import BaseModel
from power_api.BaseApi import BaseApiCall


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
    data: dict[str, EnergiDataInstance] # key = hourUTC

    def call_api(self):
        
        return super().call_api()


