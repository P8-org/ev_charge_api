import json
from pydantic import BaseModel, Json, computed_field
import requests 
import datetime


class RequestDetail(BaseModel):
    '''
    Members format
    ----------
    startDate: "2022-05-01", "now", "now-P1D"/"now%2P1D"(1 day before/after now, can be D/M/Y), "StartOfDay", "StartOfMonth", "StartOfYear"
    endDate: (optional) same as startDate
    dataset: any from energidataservice.dk
    optional: (optional) as one string, e.g. "HourUTC,PriceArea"
    filter_json: (optional) in json format, e.g. "{'PriceArea':['DK1','DK2']}"
    sort: in one string, e.g. "HourUTC desc, PriceArea"
    offset: (optional) any integer, default=0
    limit: (optional) any integer, default=100
    '''
    startDate: datetime.date | str
    endDate: datetime.date | str = ""
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

    @computed_field
    @property
    def TotalPriceDKK(self) -> float:
        return (self.SpotPriceDKK / 1000 + self.TAX + self._transportFee()) * self.VAT

    # tallene kommer fra https://elberegner.dk/elpriser-time-for-time/
    # inkluderer: spot pris, elafgift, transport af strøm, moms
    # inklurerer IKKE: abonnement/udgifter fra elselskab fordi det er meget forskelligt
    TAX: float = 0.72
    VAT: float = 1.25
    def _transportFee(self) -> float:
        energinet = 0.061
        hour = int(self.HourDK[11:13])
        if hour < 6:
            return 0.0867 + energinet
        if hour < 17:
            return 0.13 + energinet
        if hour < 21:
            return 0.338 + energinet
        return 0.13 + energinet

class EnergiData:
    # def __init__(self):
    #     self.data = []

    # data: list[EnergiDataInstance]

    def process_request(self, rd: RequestDetail):

        request_string = f'{rd.dataset}?start={rd.startDate}'

        if not rd.endDate == "":
            request_string += f'&end={rd.endDate}'
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
        else:
            request_string += '&limit=100'

        return request_string

    def __parse_request(self, r = requests.Response):
        data: list[EnergiDataInstance] = []
        r_json = r.json()

        for record in r_json["records"]:
            j = json.dumps(record)
            edi = EnergiDataInstance.model_validate_json(j)
            data.append(edi)

        return data

    def call_api(self, rd: RequestDetail) -> list[EnergiDataInstance]:
        base_url = "https://api.energidataservice.dk/dataset/"
        request_string = self.process_request(rd)

        r = requests.get(base_url+request_string)

        return self.__parse_request(r)
    
    # could make generic method... but too much work
    def get_today(self, rd: RequestDetail) -> list[EnergiDataInstance]:
        base_url = "https://api.energidataservice.dk/dataset/"
        rd.startDate = "StartOfDay"
        rd.endDate = ""

        request_string = self.process_request(rd)

        r = requests.get(base_url+request_string)

        return self.__parse_request(r)

    # could make generic method... but too much work
    def get_current_month(self, rd: RequestDetail) -> list[EnergiDataInstance]:
        base_url = "https://api.energidataservice.dk/dataset/"
        rd.startDate = "StartOfMonth"
        rd.endDate = ""

        request_string = self.process_request(rd)

        r = requests.get(base_url+request_string)
        print(base_url+request_string)
        return self.__parse_request(r)

    # could make generic method... but too much work
    def get_current_year(self, rd: RequestDetail) -> list[EnergiDataInstance]:
        base_url = "https://api.energidataservice.dk/dataset/"
        rd.startDate = "StartOfYear"
        rd.endDate = ""

        request_string = self.process_request(rd)

        r = requests.get(base_url+request_string)

        print(base_url+request_string)
        return self.__parse_request(r)

