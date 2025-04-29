import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from database.db import get_db
from models.models import CarModel, Constraint, State, UserEV, Schedule
from modules.rl_short_term_scheduling import generate_schedule
import enum


router = APIRouter()

class EvCreate(BaseModel):
    name: str
    car_model_id: int
    battery_level: float
    max_charging_power: float | None = None


def simulate_charging(ev: UserEV):
    try:
        schedule_data = list(map(float, ev.schedule.schedule_data.split(", ")))
    except: #if empty schedule_data
        return
    if ev.schedule.end < datetime.datetime.now():
        for val in schedule_data:
            ev.current_charge += val
        return
    now_hour = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    hour_idx = round((now_hour - ev.schedule.start).total_seconds() // 3600) if ev.schedule.start else None
    if hour_idx is not None and hour_idx >= 0 and hour_idx < ev.schedule.num_hours:
        ev.current_charging_power = schedule_data[hour_idx]
        if ev.current_charging_power != 0:
            ev.state = "charging"
        else:
            ev.state = "idle"
        for i in range(hour_idx):
            ev.current_charge += schedule_data[i]
        elapsed_fraction = (datetime.datetime.now().minute * 60 + datetime.datetime.now().second) / 3600
        ev.current_charge += schedule_data[hour_idx] * elapsed_fraction


@router.post("/evs")
async def create_ev(ev_create: EvCreate, db: Session = Depends(get_db)):
    ev = UserEV()
    ev.user_set_name = ev_create.name
    ev.current_charge = ev_create.battery_level
    ev.current_charging_power = 0
    ev.state = State.DISCONNECTED

    ev.car_model = db.query(CarModel).get(ev_create.car_model_id)

    ev.max_charging_power = ev.car_model.max_charging_power if ev_create.max_charging_power is None else ev_create.max_charging_power

    constraint = Constraint()
    constraint.charged_by = datetime.datetime.now() + datetime.timedelta(days=1)
    constraint.target_percentage = 0.8

    schedule = Schedule()


    ev.constraint = constraint
    ev.schedule = schedule
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev

@router.get("/evs")
async def get_evs(db: Session = Depends(get_db)):
    evs: list[UserEV] = db.query(UserEV).options(
        joinedload(UserEV.constraint),
        joinedload(UserEV.schedule),
        joinedload(UserEV.car_model)
    ).all()
    [simulate_charging(e) for e in evs]
    return evs

@router.get("/evs/{id}")
async def get_ev_by_id(id: int, db: Session = Depends(get_db)):
    ev: UserEV = db.query(UserEV).options(
        joinedload(UserEV.constraint),
        joinedload(UserEV.schedule),
        joinedload(UserEV.car_model)
    ).get(id)

    if not ev:
        raise HTTPException(status_code=404, detail="EV not found")
    
    simulate_charging(ev)

    return ev



@router.delete("/evs/{id}")
async def delete_ev_by_id(id: int, db: Session = Depends(get_db)):
    ev = db.query(UserEV).filter(UserEV.id == id).first()

    if not ev:
        raise HTTPException(status_code=404, detail="EV not found")

    db.delete(ev)
    db.commit()
    return {"detail": "EV deleted successfully"}

@router.put("/evs/{id}")
async def put_ev_by_id(id: int, ev_create: EvCreate, db: Session = Depends(get_db)): # IGNORES 'car_model_id' wether its provided or not
    ev_id = db.query(UserEV).filter(UserEV.id == id).update(
        {
            "user_set_name":ev_create.name,
            "current_charge": ev_create.battery_level,
        }
    )

    if not ev_id:
        raise HTTPException(status_code=404, detail=f"EV with given id '{id}' was not found")
    
    db.commit()
    return {"detail": f"EV with id: '{id}' updated successfully"}

