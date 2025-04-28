import requests
import urllib.request
from rich import print
from bs4 import BeautifulSoup
from requests.api import request
from pydantic import BaseModel
import json

curves : list[dict] = []

url = "https://evkx.net/models/bmw/i4/i4_edrive35/chargingcurve/"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')

tbody = soup.find_all('tr')
# print(tbody[len(tbody)-101:][0])
tr_tags = tbody[len(tbody)-101:] # -101 to get all 100 percentages


for td in tr_tags:
    td_tags = td.find_all('td')
    curve = {    
        "state_of_charge_percentage": int(td_tags[0].text.split(" ")[0]),
        "charge_speed_kW":  int(td_tags[1].text.split(" ")[0]),
        # "time_to_charge": int(td_tags[2].text.split(" ")[0]), # probably not needed
        "energy_charged_kWh": float(td_tags[3].text.split(" ")[0])
    }
    curves.append(curve)

# for curve in curves:
#     print(json.dumps(curve))
with open('charging_curve.json', 'w', encoding='utf-8') as f:
    json.dump(curves, f, ensure_ascii=False, indent=4)
