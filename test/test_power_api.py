import json
import time
from apis.EnergiData import EnergiData, EnergiDataInstance, RequestDetail
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
    control = EnergiData()

    rd_cpy = rd

    base = EnergiData()
    base.call_api(rd)  

    rd_cpy.limit = 10
    control.call_api(rd_cpy)
    assert len(base.data) != len(control.data) # just in case
    assert len(control.data) == 10

    control.data = []
    assert control.data == []
    
    base = EnergiData()
    base.call_api(rd_cpy)
    time.sleep(1)

    rd_cpy.sort_data = "SpotPriceDKK"
    control.call_api(rd_cpy)
    assert base.data != control.data
    

    control.data = []
    assert control.data == []
    rd_cpy.sort_data = ""
    assert rd_cpy.sort_data == ""
    rd_cpy.limit = 100
    assert rd_cpy.limit == 100
    time.sleep(1)
    
    rd_cpy.optional = "HourUTC,PriceArea"
    control.call_api(rd_cpy)
    assert len(control.data) > 0
    assert control.data[0].HourUTC != ""
    assert control.data[0].HourDK == ""
    assert control.data[0].PriceArea != ""
    assert control.data[0].SpotPriceDKK == 0
    assert control.data[0].SpotPriceEUR == 0

    control.data = []
    assert control.data == []
    rd_cpy.optional = ""
    assert rd_cpy.optional == ""
    time.sleep(1)
    

    rd_cpy.filter_json = json.dumps({"PriceArea": ["DK1"]})
    control.call_api(rd_cpy)
    assert control.data[0].PriceArea == "DK1"
    # control.data = []
    # rd_cpy.filter_json = json.dumps({"PriceArea": ["DK2"]})
    # control.call_api(rd_cpy)
    # assert control.data[0].PriceArea == "DK2"

    control.data = []
    assert control.data == []
    rd_cpy.filter_json = ""
    assert rd_cpy.filter_json == ""
    time.sleep(1)


    rd_cpy.offset = 3

















