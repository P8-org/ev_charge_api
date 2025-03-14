import json
from power_api.EnergiData import EnergiData, EnergiDataInstance, RequestDetail
import requests
import datetime


rd = RequestDetail(
    startDate=datetime.date(year=2024,month=4,day=17),
    endDate=datetime.date(year=2024,month=4,day=18),
    dataset="Elspotprices",
    filter_json=json.dumps({"PriceArea": ["DK1"]}),
    limit=1
)

def test_api_call():
    edi = EnergiDataInstance(
        HourUTC="2024-04-17T21:00:00",
        HourDK="2024-04-17T23:00:00",
        PriceArea="DK1",
        SpotPriceDKK=655.109985,
        SpotPriceEUR=87.809998
    )
    assert EnergiDataInstance.model_validate(edi) # does this even do anything?

    r = requests.get(f'https://api.energidataservice.dk/dataset/{rd.dataset}?{rd.dataset}?start={rd.startDate}&end={rd.endDate}&filter={rd.filter_json}&limit={rd.limit}') 
    assert r.status_code == 200

    r_json = json.dumps(r.json()["records"][0])
    r_edi = EnergiDataInstance.model_validate_json(r_json)

    assert r_edi == edi

    
def test_api_to_data():
    edi = EnergiDataInstance(
        HourUTC="2024-04-17T21:00:00",
        HourDK="2024-04-17T23:00:00",
        PriceArea="DK1",
        SpotPriceDKK=655.109985,
        SpotPriceEUR=87.809998
    )
    assert EnergiDataInstance.model_validate(edi) # does this even do anything?

    e = EnergiData()
    e.call_api(rd)

    assert edi == e.data[0]

def test_api_parameters():
    pass














