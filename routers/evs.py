import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from database.base import Base
from database.db import get_db
from models.models import CarModel, Constraint, UserEV, Schedule
from modules.rl_short_term_scheduling import generate_schedule


router = APIRouter()

class EvCreate(BaseModel):
    name: str
    car_mode_id: int
    battery_level: float

@router.post("/evs")
async def create_ev(ev_create: EvCreate, db: Session = Depends(get_db)):
    ev = UserEV()
    ev.user_set_name = ev_create.name
    ev.current_charge = ev_create.battery_level
    ev.current_charging_power = 0

    ev.car_model = db.query(CarModel).get(ev_create.car_mode_id)

    constraint = Constraint()
    constraint.charged_by = datetime.datetime.now()
    constraint.target_percentage = 0.8

    schedule = Schedule()
    schedule.end = datetime.datetime.now()
    schedule.start = datetime.datetime.now()
    schedule.schedule_data = ""
    schedule.num_hours = 0

    ev.constraint = constraint
    ev.schedule = schedule
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev

@router.get("/evs")
async def get_evs(db: Session = Depends(get_db)):
    evs = db.query(UserEV).options(
        joinedload(UserEV.constraint),
        joinedload(UserEV.schedule),
        joinedload(UserEV.car_model)
    ).all()
    return evs

@router.get("/evs/{id}")
async def get_ev_by_id(id: int, db: Session = Depends(get_db)):
    ev = db.query(UserEV).options(
        joinedload(UserEV.constraint),
        joinedload(UserEV.schedule),
        joinedload(UserEV.car_model)
    ).filter(UserEV.id == id).first()

    if not ev:
        raise HTTPException(status_code=404, detail="EV not found")
    return ev

@router.delete("/evs/{id}")
async def delete_ev_by_id(id: int, db: Session = Depends(get_db)):
    ev = db.query(UserEV).filter(UserEV.id == id).first()

    if not ev:
        raise HTTPException(status_code=404, detail="EV not found")

    db.delete(ev)
    db.commit()
    return {"detail": "EV deleted successfully"}