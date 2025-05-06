from RL.DQN.DQN import run_dqn, train_dqn
# from RL.DQN.DQN1 import run_dqn, train_dqn
from apis.EnergiData import RequestDetail
import json

#TODO MAKE THIS RUN SOMEWHERE ELSE AND DELETE
#Only created since root folder should be in path(python is weird)

cars = [
    {'id': 0, 'charge_percentage': 0, 'min_percentage': 50, 'charge': 0, 'max_charge_kw': 80, 'charge_speed': 22, 'constraints': {}},
    # {'id': 0, 'charge_percentage': 0, 'min_percentage': 80, 'charge': 0, 'max_charge_kw': 80, 'charge_speed': 22, 'constraints': {"start": 15, "end": 17}},
    # {'id': 0, 'charge_percentage': 0, 'min_percentage': 80, 'charge': 0, 'max_charge_kw': 80, 'charge_speed': 22, 'constraints': {"start": 14, "end": 17}},
    # {'id': 0, 'charge_percentage': 0, 'min_percentage': 80, 'charge': 0, 'max_charge_kw': 80, 'charge_speed': 22, 'constraints': {"start": 12, "end": 17}},
    # {'id': 0, 'charge_percentage': 0, 'min_percentage': 80, 'charge': 0, 'max_charge_kw': 80, 'charge_speed': 22, 'constraints': {"start": 7, "end": 17}},
    # {'id': 0, 'charge_percentage': 0, 'min_percentage': 0, 'charge': 40, 'max_charge_kw': 60, 'charge_speed': 22, 'constraints': {"start": 2, "end": 5}},
    # {'id': 0, 'charge_percentage': 0, 'min_percentage': 0, 'charge': 0,  'max_charge_kw': 60, 'charge_speed': 10, 'constraints': {"start":12}},
    # {'id': 0, 'charge_percentage': 0, 'min_percentage': 0, 'charge': 0,  'max_charge_kw': 60, 'charge_speed': 10, 'constraints': {"start":22}},
    # {'id': 0, 'charge_percentage': 0, 'min_percentage': 0, 'charge': 20, 'max_charge_kw': 80, 'charge_speed': 15, 'constraints': {"end":14}},
    # {'id': 0, 'charge_percentage': 0, 'min_percentage': 0, 'charge': 0,  'max_charge_kw': 60, 'charge_speed': 22, 'constraints': {}},
    # {'id': 0, 'charge_percentage': 0, 'min_percentage': 0, 'charge': 20, 'max_charge_kw': 60, 'charge_speed': 12, 'constraints': {}},
]
num_chargers = len(cars)
# num_episodes = 500
num_episodes = 1000
# num_episodes = 2000
rd = RequestDetail(
    startDate="2023-12-31T13:00",
    endDate="2024-12-31T12:00",
    # startDate="2024-04-30T13:00",
    # endDate="2024-12-31T13:00",
    dataset="Elspotprices",
    # optional="HourDK,SpotPriceDKK",
    sort_data="HourDK ASC",
    filter_json=json.dumps({"PriceArea": ["DK1"]}),
    limit=24*5, # Default=0, to limit set to a minimum of 72 hours
    offset=0
)
train_dqn(cars=cars,rd=rd,num_chargers=num_chargers,num_episodes=num_episodes)

rd = RequestDetail(
    startDate="2025-05-04T13:00",
    endDate="2025-05-05T13:00",
    dataset="Elspotprices",
    # optional="HourDK,SpotPriceDKK",
    sort_data="HourDK ASC",
    filter_json=json.dumps({"PriceArea": ["DK1"]}),
    limit=24*0, # Default=0, to limit set to a minimum of 72 hours
    offset=0
)

run_dqn(car=cars[0],rd=rd)



