import datetime
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


def simulate_charging(ev: UserEV):
    now = datetime.datetime.now()

    # Check if the schedule has ended
    if not ev.schedule or ev.schedule.end < now:
        constraint: Constraint = ev.get_previous_constraint()
        if constraint:
            ev.state = State.DISCONNECTED
            ev.current_charge = constraint.target_percentage * ev.car_model.battery_capacity
        return

    # Parse schedule data
    try:
        schedule_data = list(map(float, ev.schedule.schedule_data.split(", ")))
    except (AttributeError, ValueError):
        ev.state = State.IDLE
        return

    # Calculate the current hour index
    if not ev.schedule.start:
        ev.state = State.IDLE
        return

    now_hour = now.replace(minute=0, second=0, microsecond=0)
    hour_idx = round((now_hour - ev.schedule.start).total_seconds() // 3600)

    if 0 <= hour_idx < ev.schedule.num_hours:
        ev.current_charging_power = schedule_data[hour_idx]
        ev.state = State.CHARGING if ev.current_charging_power > 0 else State.IDLE

        # Update charge for completed hours
        ev.current_charge += sum(schedule_data[:hour_idx])

        # Add charge for the current hour based on elapsed time
        elapsed_fraction = (now.minute * 60 + now.second) / 3600
        ev.current_charge += schedule_data[hour_idx] * elapsed_fraction
    else:
        ev.state = State.IDLE


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

