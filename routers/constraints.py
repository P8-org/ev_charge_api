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
    startTime: datetime.datetime = datetime.datetime.now()
    deadline: datetime.datetime = datetime.datetime.now() + datetime.timedelta(days=1)
    target_percentage: float = 0.8


@router.post("/evs/{ev_id}/constraints")
async def edit_constraint(ev_id: int, form: ConstraintForm, db: Session = Depends(get_db)):
    ev: UserEV = db.query(UserEV).options(
        joinedload(UserEV.constraints),
        joinedload(UserEV.schedule)
    ).get(ev_id)

    if ev is None:
        raise HTTPException(status_code=404, detail=f"EV with id {ev_id} not found")

    start_time = form.startTime
    deadline = form.deadline
    target_percentage = form.target_percentage

    if target_percentage < 0 or target_percentage > 1:
        raise HTTPException(status_code=400, detail="Target percentage must be between 0.0 and 1.0")
    if deadline < datetime.datetime.now():
        raise HTTPException(status_code=400, detail="Deadline must be in the future")
    if start_time >= deadline:
        raise HTTPException(status_code=400, detail="Starttime must be before deadline")

    if form.id is not None:
        # Try to find existing constraint
        constraint = db.query(Constraint).filter_by(id=form.id, ev_id=ev_id).first()
        if constraint:
            constraint.start_time = form.startTime
            constraint.charged_by = form.deadline
            constraint.target_percentage = form.target_percentage
            db.commit()
            return {"detail": f"Constraint {constraint.id} updated"}

    new_constraint = Constraint(
        start_time=start_time,
        charged_by=deadline,
        target_percentage=target_percentage,
        ev_id=ev.id
    )
    db.add(new_constraint)
    db.commit()
    db.refresh(new_constraint)

    return {"detail": f"Constraint {new_constraint.id} created"}

    # db.commit()
    # await make_schedule(ev_id, db)

# Føler det burde være en post med ("/constraints/{id}/delete) men idk
@router.delete("/constraints/{constraint_id}")
async def delete_constraint(constraint_id: int, db: Session = Depends(get_db)):
    constraint = db.query(Constraint).filter_by(id=constraint_id).first()
    if constraint is None:
        raise HTTPException(status_code=404, detail="Constraint not found")
    db.delete(constraint)
    db.commit()
    return {"detail": f"Constraint {constraint_id} deleted"}
