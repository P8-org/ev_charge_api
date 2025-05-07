import datetime
import json
import math
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from apis.EnergiData import EnergiData, RequestDetail
from modules.benchmark_prices import Benchmark
from modules.linear_optimization_controller import adjust_rl_schedule
from modules.rl_short_term_scheduling import generate_schedule

from database.base import Base
from database.db import get_db
from models.models import UserEV

router = APIRouter()

@router.post("/schedules/generate/{ev_id}")
async def make_schedule(ev_id: int, db: Session = Depends(get_db)):
    ev: UserEV = db.query(UserEV).options(
        joinedload(UserEV.constraints),
        joinedload(UserEV.schedule),
        joinedload(UserEV.car_model)
    ).get(ev_id)

    if ev is None:
        raise HTTPException(status_code=404, detail=f"EV with id {ev_id} not found")

    duration: datetime.timedelta = ev.constraints.charged_by - datetime.datetime.now()
    num_hours = math.ceil(duration.total_seconds() / 60 / 60)
    if num_hours <= 0: raise HTTPException(status_code=400, detail="Charging duration is negative")
    target_kwh = ev.constraints.target_percentage * ev.car_model.battery_capacity
    e = EnergiData()
    formatted_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M")
    rd = RequestDetail(startDate=formatted_time, dataset="Elspotprices", filter_json=json.dumps({"PriceArea": ["DK1"]}), sort_data="HourDK ASC")
    response = e.call_api(rd)
    while datetime.datetime.fromisoformat(response[0].HourDK) < datetime.datetime.now():
        response.pop(0)

    hour_dk = [record.HourDK for record in response]
    prices = [record.SpotPriceDKK / 1000 for record in response]

    max_power = min(ev.max_charging_power, ev.car_model.max_charging_power)

    schedule = generate_schedule(num_hours, ev.current_charge, target_kwh, max_power, prices, False)
    try:
        schedule = adjust_rl_schedule(schedule, target_kwh - ev.current_charge, max_power)
        ev.schedule.feasible = True
    except:
        ev.schedule.feasible = False

    schedule = [0 if abs(x) < 1e-4 else x for x in schedule] #round very small numbers to 0

    ev.schedule.num_hours = len(schedule)
    ev.schedule.schedule_data = ", ".join(map(str, schedule))
    ev.schedule.start = (datetime.datetime.now() + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    ev.schedule.end = ev.schedule.start + datetime.timedelta(hours=ev.schedule.num_hours)
    ev.schedule.start_charge = ev.current_charge

    b = Benchmark(schedule,prices, target_kwh - ev.current_charge, max_power)
    ev.schedule.price = b.optimized_schedule_price()
    ev.schedule.greedy_price = b.greedy_schedule_price()

    db.commit()

    return [{"time": h, "price": p, "charging": b} for h, p, b in zip(hour_dk, prices, schedule)]