import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload
from database.db import get_db
from models.models import UserEV
from routers.schedules import make_schedule


router = APIRouter()

class ConstraintForm(BaseModel):
    start_time: datetime.datetime = datetime.datetime.now()
    deadline: datetime.datetime = datetime.datetime.now() + datetime.timedelta(days=1)
    target_percentage: float = 0.8


@router.post("/evs/{ev_id}/constraints")
async def edit_constraint(ev_id: int, form: ConstraintForm, db: Session = Depends(get_db)):
    print(form)
    ev: UserEV = db.query(UserEV).options(
        joinedload(UserEV.constraint),
        joinedload(UserEV.schedule)
    ).get(ev_id)

    if ev is None:
        raise HTTPException(status_code=404, detail=f"EV with id {ev_id} not found")

    start_time = form.start_time
    deadline = form.deadline
    target_percentage = form.target_percentage

    if target_percentage < 0 or target_percentage > 1:
        raise HTTPException(status_code=400, detail="Target percentage must be between 0.0 and 1.0")
    if deadline < datetime.datetime.now():
        raise HTTPException(status_code=400, detail="Deadline must be in the future")
    if start_time >= deadline:
        raise HTTPException(status_code=400, detail="Starttime must be before deadline")

    ev.constraint.starttime = start_time
    ev.constraint.charged_by = deadline
    ev.constraint.target_percentage = target_percentage

    db.commit()
    await make_schedule(ev_id, db)
