import json
import time
from apis.EnergiData import EnergiData, EnergiDataInstance, RequestDetail
import requests
import datetime


rd = RequestDetail(
    startDate=datetime.datetime(year=2024,month=4,day=17),
    endDate=datetime.datetime(year=2024,month=4,day=18),
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
    
    assert edi == e.call_api(rd)[0]

def test_api_parameters():
    control = EnergiData()

    rd_cpy = rd

    base = EnergiData()
    base_data = base.call_api(rd)  

    rd_cpy.limit = 10
    control_data = control.call_api(rd_cpy)
    assert len(base_data) != len(control_data) # just in case
    assert len(control_data) == 10

    control_data = []
    assert control_data == []

    base = EnergiData()
    base_data = base.call_api(rd_cpy)
    # time.sleep(1)

    rd_cpy.sort_data = "SpotPriceDKK"
    data = control.call_api(rd_cpy)
    assert base_data != control_data


    control_data = []
    assert control_data == []
    rd_cpy.sort_data = ""
    assert rd_cpy.sort_data == ""
    rd_cpy.limit = 100
    assert rd_cpy.limit == 100
    # time.sleep(1)

    rd_cpy.optional = "HourUTC,PriceArea"
    control_data = control.call_api(rd_cpy)
    assert len(control_data) > 0
    assert control_data[0].HourUTC != ""
    assert control_data[0].HourDK == ""
    assert control_data[0].PriceArea != ""
    assert control_data[0].SpotPriceDKK == 0
    assert control_data[0].SpotPriceEUR == 0

    control_data = []
    assert control_data == []
    rd_cpy.optional = ""
    assert rd_cpy.optional == ""
    # time.sleep(1)


    rd_cpy.filter_json = json.dumps({"PriceArea": ["DK1"]})
    control_data = control.call_api(rd_cpy)
    assert control_data[0].PriceArea == "DK1"
    # control_data = []
    # rd_cpy.filter_json = json.dumps({"PriceArea": ["DK2"]})
    # control.call_api(rd_cpy)
    # assert control_data[0].PriceArea == "DK2"

    control_data = []
    assert control_data == []
    rd_cpy.filter_json = ""
    assert rd_cpy.filter_json == ""
    # time.sleep(1)


    rd_cpy.offset = 3

