import datetime
import math
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from database.db import get_db
from models.models import CarModel, Constraint, State, UserEV, Schedule


router = APIRouter()

class EvCreate(BaseModel):
    name: str
    car_model_id: int
    battery_level: float
    max_charging_power: float | None = None


def simulate_charging(ev: UserEV, now: datetime.datetime = datetime.datetime.now()):
    ev.current_charging_power = 0
    if not ev.schedule:
        ev.state = State.DISCONNECTED
        return
    
    schedule_data = list(map(float, ev.schedule.schedule_data.split(", ")))
    num_hours = len(schedule_data)
    start = ev.schedule.start
    end = ev.schedule.end
    ev.current_charge = ev.schedule.start_charge
    
    # handle before, and after, and at beginning
    if now < start:
        return
    elif now >= end:
        ev.current_charge = ev.schedule.start_charge + sum(schedule_data)
        return
    elif now == start:
        ev.current_charging_power = ev.max_charging_power
        return
    
    
    # handle single hour
    if num_hours == 1:
        if schedule_data[0] == 0: return
        length = (end - start).total_seconds()
        start_to_now = (now - start).total_seconds()
        fraction = start_to_now / length
        ev.current_charge = ev.schedule.start_charge + schedule_data[0] * fraction
        ev.current_charging_power = ev.max_charging_power
        return

    # handle multiple hours
    current_hour = math.ceil((now - ev.schedule.start.replace(minute=0, second=0, microsecond=0)).total_seconds() / 3600)

    for i in range(current_hour):
        # set charging power
        if schedule_data[i] == 0:
            ev.current_charging_power = 0
        else:
            ev.current_charging_power = ev.max_charging_power

        if i == 0: # first hour
            if now.hour == start.hour:
                next_hour = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                length = (next_hour - start).total_seconds()
                start_to_now = (now - start).total_seconds()
                fraction = start_to_now / length
                ev.current_charge += schedule_data[i] * fraction
            else:
                ev.current_charge += schedule_data[i]
                
        elif i == num_hours-1: # last hour
            hour_zeroed = now.replace(minute=0, second=0, microsecond=0)
            length = (end - hour_zeroed).total_seconds()
            start_to_now = (now - hour_zeroed).total_seconds()
            fraction = start_to_now / length
            ev.current_charge += schedule_data[i] * fraction
            

        elif i == current_hour-1: # current hour
            hour_zeroed = now.replace(minute=0, second=0, microsecond=0)
            length = 3600
            start_to_now = (now - hour_zeroed).total_seconds()
            fraction = start_to_now / length
            if fraction == 0: fraction = 1
            ev.current_charge += schedule_data[i] * fraction

        else: # any other hour
            ev.current_charge += schedule_data[i]
    



@router.post("/evs")
async def create_ev(ev_create: EvCreate, db: Session = Depends(get_db)):
    ev = UserEV()
    ev.user_set_name = ev_create.name
    ev.current_charge = ev_create.battery_level
    ev.current_charging_power = 0
    ev.state = State.DISCONNECTED

    ev.car_model = db.query(CarModel).get(ev_create.car_model_id)

    # ev.max_charging_power = ev.car_model.max_charging_power if ev_create.max_charging_power is None else ev_create.max_charging_power
    ev.max_charging_power = 11
    
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev

@router.get("/evs")
async def get_evs(db: Session = Depends(get_db)):
    evs: list[UserEV] = db.query(UserEV).options(
        joinedload(UserEV.constraints),
        joinedload(UserEV.schedule),
        joinedload(UserEV.car_model)
    ).all()
    db.expunge_all()
    [simulate_charging(e) for e in evs]
    return evs

@router.get("/evs/{id}")
async def get_ev_by_id(id: int, db: Session = Depends(get_db)):
    ev: UserEV = db.query(UserEV).options(
        joinedload(UserEV.constraints),
        joinedload(UserEV.schedule),
        joinedload(UserEV.car_model),
    ).get(id)
    db.expunge_all()

    if not ev:
        raise HTTPException(status_code=404, detail="EV not found")
    
    simulate_charging(ev)
    return ev



@router.delete("/evs/{id}")
async def delete_ev_by_id(id: int, db: Session = Depends(get_db)):
    ev = db.query(UserEV).filter(UserEV.id == id).first()

    if not ev:
        raise HTTPException(status_code=404, detail="EV not found")
    
    if ev.schedule:
        db.delete(ev.schedule)

    db.delete(ev)
    db.commit()
    return {"detail": "EV deleted successfully"}

@router.put("/evs/{id}")
async def put_ev_by_id(id: int, ev_create: EvCreate, db: Session = Depends(get_db)):
    ev_id = db.query(UserEV).filter(UserEV.id == id).update(
        {
            UserEV.user_set_name: ev_create.name,
            UserEV.car_model_id: ev_create.car_model_id,
            UserEV.current_charge: ev_create.battery_level,
            UserEV.max_charging_power: ev_create.max_charging_power
        }
    )

    if not ev_id:
        raise HTTPException(status_code=404, detail=f"EV with given id '{id}' was not found")
    
    db.commit()
    return {"detail": f"EV with id: '{id}' updated successfully"}

