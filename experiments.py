import datetime
import json
import random
import numpy as np

import requests
from collections import defaultdict
from apis.EnergiData import EnergiData, RequestDetail
from models.models import CarModel, UserEV
from modules.benchmark_prices import Benchmark
from modules.linear_optimization_controller import adjust_rl_schedule, optimize_charging_schedule_unused
from modules.rl_short_term_scheduling import generate_schedule

from RL.DQN.DQN_single import run_dqn

def is_weekday(datetime: datetime.datetime) -> bool:
    return datetime.weekday() < 5


def simulate(car: UserEV, target_percentage: float, prices: list[float], times: list[datetime]) -> list[float]:
    target_kwh = target_percentage * car.car_model.battery_capacity
    schedule_data = generate_schedule(num_hours=len(prices), initial_soc=car.current_charge, battery_capacity=target_kwh, max_chargin_rate=car.max_charging_power, prices=prices)
    schedule_data = adjust_rl_schedule(schedule_data, target_kwh - car.current_charge, car.max_charging_power)  
    return schedule_data

def simulate_dqn(car: UserEV, target_percentage: float, prices: list[float], times: list[datetime]) -> list[float]:
    target_kwh = target_percentage * car.car_model.battery_capacity
    c = {
        'id': 1, 
        'charge_percentage': min(car.current_charge / car.car_model.battery_capacity * 100, 100),
        'min_percentage': target_percentage * 100,
        'charge': car.current_charge,
        'max_charge_kw': car.car_model.battery_capacity,
        'charge_speed': car.max_charging_power,
        'constraints': {}
    }
    schedule_data : list[float] = run_dqn(c, prices, times)[0]['charge_kw']
    schedule_data = adjust_rl_schedule(schedule_data, target_kwh - car.current_charge, car.max_charging_power)  
    return schedule_data

def simulate_lp(car: UserEV, target_percentage: float, prices: list[float], times: list[datetime]) -> list[float]:
    target_kwh = target_percentage * car.car_model.battery_capacity
    schedule_data = optimize_charging_schedule_unused(prices, target_kwh, car.current_charge, car.max_charging_power, 1, len(prices))
    return schedule_data

def get_co2_data(start_date, end_date) -> list[float]:
    rd = RequestDetail(
        startDate=start_date,
        endDate=end_date,
        dataset="CO2EmisProg",
        sort_data="Minutes5DK%20ASC",
        filter_json=json.dumps({"PriceArea": ["DK1"]}),
        limit=0
    )
    base_url = "https://api.energidataservice.dk/dataset/"
    request_string = e.process_request(rd)
    co2_data = requests.get(base_url+request_string).json()

    # co2 data is in 15 minutes resolution. We need 1 hour resolution, so we find the avg. for each hour
    hourly_co2_data = defaultdict(list)

    for record in co2_data["records"]:
        # Extract the timestamp and CO2 emission value
        timestamp = datetime.datetime.fromisoformat(record['Minutes5DK'])
        hour_key = timestamp.replace(minute=0, second=0, microsecond=0)  # Round to the hour
        hourly_co2_data[hour_key].append(record['CO2Emission'])

    sorted_hourly_co2_data = dict(sorted(hourly_co2_data.items()))

    hourly_co2_avg = [sum(values) / len(values) for hour, values in sorted_hourly_co2_data.items()]
    return hourly_co2_avg

car = UserEV()
car.max_charging_power = 11
car.car_model = CarModel(model_name="Test car", battery_capacity=100, max_charging_power=11, model_year=2025)
start_date = "2024-05-12T00:00"
end_date = "2025-05-13T00:00"


e = EnergiData()
rd = RequestDetail(
    startDate=start_date,
    endDate=end_date,
    dataset="Elspotprices",
    sort_data="HourDK ASC",
    filter_json=json.dumps({"PriceArea": ["DK1"]}),
    limit=0
)
data = e.call_api(rd)


co2_data = get_co2_data(start_date, end_date)


days = int(len(data) / 24)

total_kwh = 0
total_kwh_dqn = 0
total_kwh_lp = 0

optimal_price = 0
greedy_price = 0
optimal_price_dqn = 0
optimal_price_lp = 0

optimal_co2 = 0
greedy_co2 = 0

random.seed(0)

charged_prev_day = True
days_charging = 0
period_days = 6
hours_per_day = 24
period_hours = period_days * hours_per_day

for day in range(0, days - period_days + 1, period_days):
    print(f"\rDay {day}", end="", flush=True)
    start_idx = day * hours_per_day
    weekday: bool = is_weekday(datetime.datetime.fromisoformat(data[start_idx].HourDK))
    start = (datetime.datetime.fromisoformat(data[start_idx].HourDK)).strftime("%Y-%m-%dT%H:%M")
    end = (datetime.datetime.fromisoformat(data[start_idx].HourDK) + datetime.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M")

    if weekday and not charged_prev_day: # normal commute. skip every second day
        days_charging += 1
        charged_prev_day = True
        car.current_charge = 62
        leave_hour = 12
        home_hour = 4

        slice_start = start_idx + home_hour
        slice_end = start_idx + period_hours + leave_hour
        day_data = data[slice_start:slice_end]
        prices = [record.TotalPriceDKK for record in day_data]
        times = [np.datetime64(record.HourDK) for record in day_data]
        
        schedule_data = simulate(car=car, target_percentage=0.8, prices=prices, times=times)
        schedule_data_dqn = simulate_dqn(car=car, target_percentage=0.8, prices=prices, times=times)
        # schedule_data = schedule_data_dqn
        schedule_data_lp = simulate_lp(car=car, target_percentage=0.8, prices=prices, times=times)
        # schedule_data = schedule_data_lp

        # differences = []
        # diff_schedule_data = []
        # diff_schedule_dqn = []
        # for i, (a, b) in enumerate(zip(schedule_data, schedule_data_dqn)):
        #     if a != b:
        #         differences.append(i)
        #         diff_schedule_data.append(a)
        #         diff_schedule_dqn.append(b) 
        # if differences != []:
        #     print(f" | Difference at index {differences}")
        # schedule_data = schedule_data_dqn
        # schedule_data = diff_schedule_data
        # schedule_data = diff_schedule_dqn

        total_kwh += sum(schedule_data)
        total_kwh_dqn += sum(schedule_data_dqn)
        total_kwh_lp += sum(schedule_data_lp)

        target_kwh = 0.8 * car.car_model.battery_capacity

        b = Benchmark(schedule_data, prices, target_kwh - car.current_charge, car.max_charging_power)
        greedy_price += b.greedy_schedule_price()
        optimal_price += b.optimized_schedule_price()
        bd = Benchmark(schedule_data_dqn, prices, target_kwh - car.current_charge, car.max_charging_power)
        optimal_price_dqn += bd.optimized_schedule_price()
        bl = Benchmark(schedule_data_lp, prices, target_kwh - car.current_charge, car.max_charging_power)
        optimal_price_lp += bl.optimized_schedule_price()

        # for i in range(len(schedule_data)):
        #     charge = schedule_data[i]
        #     optimal_co2 += co2_data[slice_start + i] * charge
        #
        # schedule_data = sorted(schedule_data, reverse=True)
        #
        # for i in range(len(schedule_data)):
        #     charge = schedule_data[i]
        #     if charge < 0.1: 
        #         continue
        #     greedy_co2 += co2_data[slice_start + i] * charge


    elif random.random() < 0.2: # dont commute if weekend, take long drive (30 kWh) 1/5 of days
        days_charging += 1
        car.current_charge = 50 if charged_prev_day else 40 # because if not charged prev day it has -10 soc
        charged_prev_day = True
        leave_hour = 12
        home_hour = 4

        slice_start = start_idx + home_hour
        slice_end = start_idx + period_hours + leave_hour
        day_data = data[slice_start:slice_end]
        prices = [record.TotalPriceDKK for record in day_data]
        times = [np.datetime64(record.HourDK) for record in day_data]

        schedule_data = simulate(car=car, target_percentage=0.8, prices=prices, times=times)
        schedule_data_dqn = simulate_dqn(car=car, target_percentage=0.8, prices=prices, times=times)
        # schedule_data = schedule_data_dqn
        schedule_data_lp = simulate_lp(car=car, target_percentage=0.8, prices=prices, times=times)
        # schedule_data = schedule_data_lp

        # differences = []
        # diff_schedule_data = []
        # diff_schedule_dqn = []
        # for i, (a, b) in enumerate(zip(schedule_data, schedule_data_dqn)):
        #     if a != b:
        #         differences.append(i)
        #         diff_schedule_data.append(a)
        #         diff_schedule_dqn.append(b) 
        # if differences != []:
        #     print(f" | Difference at index {differences}")
        # schedule_data = diff_schedule_data
        # schedule_data = diff_schedule_dqn

        total_kwh += sum(schedule_data)
        total_kwh_dqn += sum(schedule_data_dqn)
        total_kwh_lp += sum(schedule_data_lp)

        target_kwh = 0.8 * car.car_model.battery_capacity

        b = Benchmark(schedule_data, prices, target_kwh - car.current_charge, car.max_charging_power)
        greedy_price += b.greedy_schedule_price()
        optimal_price += b.optimized_schedule_price()
        bd = Benchmark(schedule_data_dqn, prices, target_kwh - car.current_charge, car.max_charging_power)
        optimal_price_dqn += bd.optimized_schedule_price()
        bl = Benchmark(schedule_data_lp, prices, target_kwh - car.current_charge, car.max_charging_power)
        optimal_price_lp += bl.optimized_schedule_price()

        # for i in range(len(schedule_data)):
        #     charge = schedule_data[i]
        #     if charge < 0.1: 
        #         continue
        #     optimal_co2 += co2_data[slice_start + i] * charge
        #
        # schedule_data = sorted(schedule_data, reverse=True)
        #
        # for i in range(len(schedule_data)):
        #     charge = schedule_data[i]
        #     if charge < 0.1: 
        #         continue
        #     greedy_co2 += co2_data[slice_start + i] * charge
    else:
        charged_prev_day = False


print("\n")
print(f"greedy price: {round(greedy_price)} kr")
print("\n")
print(f"[QL] optimal price: {round(optimal_price)} kr")
print(f"[QL] savings: {round((greedy_price - optimal_price) / greedy_price * 100)}%")
print(f"[QL] total kwh used: {total_kwh} kwh")
print("\n")
print(f"[DQL]optimal price: {round(optimal_price_dqn)} kr")
print(f"[DQL]savings: {round((greedy_price - optimal_price_dqn) / greedy_price * 100)}%")
print(f"[DQL]total kwh used: {total_kwh_dqn} kwh")
print("\n")
print(f"[LP] optimal price: {round(optimal_price_lp)} kr")
print(f"[LP] savings: {round((greedy_price - optimal_price_lp) / greedy_price * 100)}%")
print(f"[LP] total kwh used: {total_kwh_lp} kwh")

print("\n")

# print(f"greedy co2: {round(greedy_co2 / 1000)} kg")
# print(f"optimal co2: {round(optimal_co2 / 1000)} kg")
# print(f"savings: {round((greedy_co2 - optimal_co2) / greedy_co2 * 100)}%")


print(f"days charged: {days_charging}")
