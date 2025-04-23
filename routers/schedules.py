import datetime
import json
import math
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from apis.EnergiData import EnergiData, RequestDetail
from modules.linear_optimization_controller import adjust_rl_schedule
from modules.rl_short_term_scheduling import generate_schedule

from database.base import Base
from database.db import get_db
from models.models import UserEV
import pytz


router = APIRouter()

@router.post("/schedules/generate/{ev_id}")
async def make_schedule(ev_id: int, db: Session = Depends(get_db)):
    ev: UserEV = db.query(UserEV).options(
        joinedload(UserEV.constraint),
        joinedload(UserEV.schedule)
    ).get(ev_id)

    if ev is None:
        raise HTTPException(status_code=404, detail=f"EV with id {ev_id} not found")

    duration: datetime.timedelta = ev.constraint.charged_by - datetime.datetime.now()
    num_hours = math.ceil(duration.total_seconds() / 60 / 60)
    if num_hours <= 0: raise HTTPException(status_code=400, detail="Charging duration is negative")
    target_kwh = ev.constraint.target_percentage * ev.car_model.battery_capacity
    e = EnergiData()
    rd = RequestDetail(startDate="now", dataset="Elspotprices", filter_json=json.dumps({"PriceArea": ["DK1"]}), sort_data="HourDK ASC")
    response = e.call_api(rd)
    
    hour_dk = [record.HourDK for record in response]
    prices = [record.SpotPriceDKK / 1000 for record in response]

    schedule = generate_schedule(num_hours, ev.current_charge, target_kwh, ev.car_model.max_charging_power, prices, False)
    schedule = adjust_rl_schedule(schedule, target_kwh - ev.current_charge, ev.car_model.max_charging_power)

    ev.schedule.num_hours = len(schedule)
    ev.schedule.schedule_data = ", ".join(map(str, schedule))
    ev.schedule.start = datetime.datetime.now()
    ev.schedule.end = ev.schedule.start + datetime.timedelta(ev.schedule.num_hours)

    db.commit()

    return [{"time": h, "price": p, "charging": b} for h, p, b in zip(hour_dk, prices, schedule)]