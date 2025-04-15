import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from database.base import Base
from database.db import get_db
from models.models import CarModel
from modules.rl_short_term_scheduling import generate_schedule


router = APIRouter()

class CarModelCreate(BaseModel):
    name: str
    year: int
    battery_capacity: float
    max_charging_power: float

@router.post("/carmodels")
async def create_car_model(form: CarModelCreate, db: Session = Depends(get_db)):
    new_model = CarModel(
        model_name=form.name,
        model_year=form.year,
        battery_capacity=form.battery_capacity,
        max_charging_power=form.max_charging_power
    )

    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    return new_model

@router.get("/carmodels")
async def get_car_models(query: str = None, db: Session = Depends(get_db)):
    if query is not None:
        models = db.query(CarModel).filter(CarModel.model_name.contains(query)).all()
    else:
        models = db.query(CarModel).all()
    return models