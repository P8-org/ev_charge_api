import datetime
import json
import random

import requests
from collections import defaultdict
from apis.EnergiData import EnergiData, RequestDetail
from models.models import CarModel, UserEV
from modules.benchmark_prices import Benchmark
from modules.linear_optimization_controller import adjust_rl_schedule
from modules.rl_short_term_scheduling import generate_schedule

def is_weekday(datetime: datetime.datetime) -> bool:
    return datetime.weekday() < 5


def simulate(car: UserEV, target_percentage: float, prices: list[float]) -> list[float]:
    target_kwh = target_percentage * car.car_model.battery_capacity
    schedule_data = generate_schedule(num_hours=len(prices), initial_soc=car.current_charge, battery_capacity=target_kwh, max_chargin_rate=car.max_charging_power, prices=prices)
    schedule_data = adjust_rl_schedule(schedule_data, target_kwh - car.current_charge, car.max_charging_power)
    
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

optimal_price = 0
greedy_price = 0

optimal_co2 = 0
greedy_co2 = 0

random.seed(0)


for day in range(days-1):
    print(f"\rDay {day}", end="", flush=True)
    start_idx = day * 24
    weekday: bool = is_weekday(datetime.datetime.fromisoformat(data[start_idx].HourDK))

    if weekday: # normal commute
        car.current_charge = 65
        leave_hour = 8
        home_hour = 17

        slice_start = start_idx + home_hour
        slice_end = start_idx + 24 + leave_hour
        day_data = data[slice_start:slice_end]
        prices = [record.TotalPriceDKK for record in day_data]

        schedule_data = simulate(car=car, target_percentage=0.8, prices=prices)

        total_kwh += sum(schedule_data)

        target_kwh = 0.8 * car.car_model.battery_capacity

        b = Benchmark(schedule_data, prices, target_kwh - car.current_charge, car.max_charging_power)
        greedy_price += b.greedy_schedule_price()
        optimal_price += b.optimized_schedule_price()

        for i in range(len(schedule_data)):
            charge = schedule_data[i]
            optimal_co2 += co2_data[slice_start + i] * charge

        schedule_data = sorted(schedule_data, reverse=True)

        for i in range(len(schedule_data)):
            charge = schedule_data[i]
            if charge < 0.1: 
                continue
            greedy_co2 += co2_data[slice_start + i] * charge


    else: # dont commute if weekend
        if random.random() < 0.2: # take long drive 1/5 of days
            car.current_charge = 40
            leave_hour = 8
            home_hour = 18

            slice_start = start_idx + home_hour
            slice_end = start_idx + 24 + leave_hour
            day_data = data[slice_start:slice_end]
            prices = [record.TotalPriceDKK for record in day_data]

            schedule_data = simulate(car=car, target_percentage=0.8, prices=prices)

            total_kwh += sum(schedule_data)

            target_kwh = 0.8 * car.car_model.battery_capacity

            b = Benchmark(schedule_data, prices, target_kwh - car.current_charge, car.max_charging_power)
            greedy_price += b.greedy_schedule_price()
            optimal_price += b.optimized_schedule_price()

            for i in range(len(schedule_data)):
                charge = schedule_data[i]
                if charge < 0.1: 
                    continue
                optimal_co2 += co2_data[slice_start + i] * charge

            schedule_data = sorted(schedule_data, reverse=True)

            for i in range(len(schedule_data)):
                charge = schedule_data[i]
                if charge < 0.1: 
                    continue
                greedy_co2 += co2_data[slice_start + i] * charge


print("\n")
print(f"greedy price: {round(greedy_price)} kr")
print(f"optimal price: {round(optimal_price)} kr")
print(f"savings: {round((greedy_price - optimal_price) / greedy_price * 100)}%")

print("\n")

print(f"greedy co2: {round(greedy_co2 / 1000)} kg")
print(f"optimal co2: {round(optimal_co2 / 1000)} kg")
print(f"savings: {round((greedy_co2 - optimal_co2) / greedy_co2 * 100)}%")

print(f"total kwh used: {total_kwh} kwh")