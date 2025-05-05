from RL.DQN.DQN import run_dqn
from apis.EnergiData import RequestDetail
import json

#TODO MAKE THIS RUN SOMEWHERE ELSE AND DELETE
#Only created since root folder should be in path(python is weird)

cars = [
    # {'id': 0, 'charged': False, 'charge': 0, 'charge_percentage': 0, 'max_charge': 80, 'charge_speed': 22, 'constraints': {}},
    # {'id': 0, 'charged': False, 'charge': 0, 'charge_percentage': 0, 'max_charge': 80, 'charge_speed': 22, 'constraints': {"start": 10, "end": 17}},
    # {'id': 0, 'charged': False, 'charge': 40, 'charge_percentage': 0, 'max_charge': 60, 'charge_speed': 22, 'constraints': {"start": 2, "end": 5}},
    # {'id': 0, 'charged': False, 'charge': 0, 'charge_percentage': 0, 'max_charge': 60, 'charge_speed': 10, 'constraints': {"start":12}},
    # {'id': 0, 'charged': False, 'charge': 20, 'charge_percentage': 0, 'max_charge': 80, 'charge_speed': 15, 'constraints': {"end":14}},
    # {'id': 0, 'charged': False, 'charge': 0, 'charge_percentage': 0, 'max_charge': 60, 'charge_speed': 22, 'constraints': {}},
    {'id': 0, 'charged': False, 'charge': 20, 'charge_percentage': 0, 'max_charge': 60, 'charge_speed': 12, 'constraints': {}},

    # {'id': 0, 'charged': False, 'charge': 0, 'charge_percentage': 0, 'max_charge': 80, 'charge_speed': 22, 'constraints': {"start": 10, "end": 17}},
    # {'id': 1, 'charged': False, 'charge': 40, 'charge_percentage': 0, 'max_charge': 60, 'charge_speed': 22, 'constraints': {"start": 2, "end": 5}},
    # {'id': 2, 'charged': False, 'charge': 0, 'charge_percentage': 0, 'max_charge': 60, 'charge_speed': 10, 'constraints': {"start":12}},
    # {'id': 3, 'charged': False, 'charge': 20, 'charge_percentage': 0, 'max_charge': 80, 'charge_speed': 15, 'constraints': {"end":14}},
    # {'id': 4, 'charged': False, 'charge': 0, 'charge_percentage': 0, 'max_charge': 60, 'charge_speed': 22, 'constraints': {}},
    # {'id': 5, 'charged': False, 'charge': 20, 'charge_percentage': 0, 'max_charge': 60, 'charge_speed': 12, 'constraints': {}},
]
num_chargers = len(cars)
# num_episodes = 500
num_episodes = 1000
# num_episodes = 2000
rd = RequestDetail(
    # startDate="2023-12-31T13:00",
    # endDate="2024-12-31T12:00",
    # startDate="2024-04-30T13:00",
    # endDate="2024-12-31T12:00",
    startDate="2025-01-01T13:00",
    # endDate="2025-12-31T12:00",
    dataset="Elspotprices",
    # optional="HourDK,SpotPriceDKK",
    sort_data="HourDK ASC",
    filter_json=json.dumps({"PriceArea": ["DK1"]}),
    limit=24*20, # Default=0, to limit set to a minimum of 72 hours
    offset=0
)
run_dqn(cars=cars,rd=rd,num_chargers=num_chargers,num_episodes=num_episodes)



