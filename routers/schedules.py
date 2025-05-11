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
from models.models import Constraint, Schedule, UserEV

router = APIRouter()

@router.post("/evs/{ev_id}/schedules/generate")
async def make_schedule(ev_id: int, db: Session = Depends(get_db)):
    ev: UserEV = db.query(UserEV).options(
        joinedload(UserEV.constraints),
        joinedload(UserEV.schedule),
        joinedload(UserEV.car_model)
    ).get(ev_id)

    if ev is None:
        raise HTTPException(status_code=404, detail=f"EV with id {ev_id} not found")
    
    constraint: Constraint = ev.get_next_constraint()

    if constraint is None:
        raise HTTPException(status_code=400, detail=f"EV with id {ev_id} has no upcoming constraints")

    duration: datetime.timedelta = constraint.end_time.replace(second=0, microsecond=0, minute=0) - constraint.start_time
    num_hours = math.floor(duration.total_seconds() / 60 / 60)
    if num_hours <= 0: raise HTTPException(status_code=400, detail="Charging duration is negative")
    target_kwh = constraint.target_percentage * ev.car_model.battery_capacity
    e = EnergiData()
    formatted_time = constraint.start_time.strftime("%Y-%m-%dT%H:%M")
    rd = RequestDetail(startDate=formatted_time, dataset="Elspotprices", filter_json=json.dumps({"PriceArea": ["DK1"]}), sort_data="HourDK ASC")
    response = e.call_api(rd)
    while datetime.datetime.fromisoformat(response[0].HourDK) < constraint.start_time:
        response.pop(0)

    hour_dk = [record.HourDK for record in response]
    prices = [record.SpotPriceDKK / 1000 for record in response]

    max_power = min(ev.max_charging_power, ev.car_model.max_charging_power)

    schedule_data = generate_schedule(num_hours, ev.current_charge, target_kwh, max_power, prices, False)
    schedule_data = adjust_rl_schedule(schedule_data, target_kwh - ev.current_charge, max_power)

    schedule_data = [0 if abs(x) < 1e-4 else x for x in schedule_data] #round very small numbers to 0

    if ev.schedule is not None:
        db.delete(ev.schedule)

    ev.schedule = Schedule()
    ev.schedule.feasible = ev.current_charge + sum(schedule_data) >= target_kwh
    ev.schedule.num_hours = len(schedule_data)
    ev.schedule.schedule_data = ", ".join(map(str, schedule_data))
    ev.schedule.start = datetime.datetime.fromisoformat(hour_dk[0])
    ev.schedule.end = ev.schedule.start + datetime.timedelta(hours=num_hours)
    ev.schedule.start_charge = ev.current_charge
    ev.schedule.constraint = constraint

    b = Benchmark(schedule_data,prices, target_kwh - ev.current_charge, max_power)
    ev.schedule.price = b.optimized_schedule_price()
    ev.schedule.greedy_price = b.greedy_schedule_price()

    db.commit()

    return {"feasible": ev.schedule.feasible , "schedule": [{"time": h, "price": p, "charging": b} for h, p, b in zip(hour_dk, prices, schedule_data)]}