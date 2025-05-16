import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload
from database.db import get_db
from models.models import UserEV, Constraint
from routers.schedules import make_schedule


router = APIRouter()

class ConstraintForm(BaseModel):
    id: int | None = None
    start_time: datetime.datetime = datetime.datetime.now()
    end_time: datetime.datetime = datetime.datetime.now() + datetime.timedelta(days=1)
    target_percentage: float = 0.8


@router.post("/evs/{ev_id}/constraints")
async def edit_constraint(ev_id: int, form: ConstraintForm, db: Session = Depends(get_db)):
    ev: UserEV = db.query(UserEV).options(
        joinedload(UserEV.constraints),
        joinedload(UserEV.schedule)
    ).get(ev_id)

    if ev is None:
        raise HTTPException(status_code=404, detail=f"EV with id {ev_id} not found")

    generate_new_schedule = False
    start_time = form.start_time
    end_time = form.end_time
    target_percentage = form.target_percentage

    if target_percentage < 0 or target_percentage > 1:
        raise HTTPException(status_code=400, detail="Target percentage must be between 0.0 and 1.0")
    if end_time < datetime.datetime.now():
        raise HTTPException(status_code=400, detail="Deadline must be in the future")
    if start_time >= end_time:
        raise HTTPException(status_code=400, detail="Starttime must be before deadline")

    if form.id is not None:
        # Try to find existing constraint
        constraint = db.query(Constraint).filter_by(id=form.id, ev_id=ev_id).first()
        if constraint:
            if ev.schedule and constraint.id == ev.schedule.constraint_id: # if editing constraint used for current schedule -> generate new schedule
                generate_new_schedule = True
            constraint.start_time = form.start_time
            constraint.end_time = form.end_time
            constraint.target_percentage = form.target_percentage
            db.commit()
            if ev.should_generate_new_schedule() or generate_new_schedule:
                print("generated schedule for id", ev_id)
                make_schedule(ev_id, db)
            return {"detail": f"Constraint {constraint.id} updated"}

    new_constraint = Constraint(
        start_time=start_time,
        end_time=end_time,
        target_percentage=target_percentage,
        ev_id=ev.id
    )
    db.add(new_constraint)
    db.commit()
    db.refresh(new_constraint)

    
    if ev.should_generate_new_schedule() or generate_new_schedule:
        print("generated schedule for id", ev_id)
        make_schedule(ev_id, db)

    return {"detail": f"Constraint {new_constraint.id} created"}


# Føler det burde være en post med ("/constraints/{id}/delete) men idk
@router.delete("/constraints/{constraint_id}")
async def delete_constraint(constraint_id: int, db: Session = Depends(get_db)):
    constraint = db.query(Constraint).filter_by(id=constraint_id).first()
    if constraint is None:
        raise HTTPException(status_code=404, detail="Constraint not found")
    if constraint.ev.schedule is not None and constraint.ev.schedule.constraint_id == constraint_id:
        db.delete(constraint.ev.schedule)
    db.delete(constraint)
    db.commit()
    return {"detail": f"Constraint {constraint_id} deleted"}
