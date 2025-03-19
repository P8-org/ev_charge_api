from fastapi import APIRouter, Depends
from pydantic import BaseModel
from requests import Session
from sqlalchemy import Column, Integer, String, Float

from database.base import Base
from database.db import get_db


router = APIRouter()

class Ev(Base):
    __tablename__ = "evs"
    id = Column(Integer, primary_key=True, index=True)
    name: str = Column(String)
    battery_capacity: float = Column(Float)
    """ Battery capacity (kwH) """
    battery_level: float = Column(Float)
    """ Current battery level (kwH) """
    max_charging_speed: float = Column(Float)
    """ Max charging speed (kw) """

class EvCreate(BaseModel):
    name: str
    battery_capacity: float
    battery_level: float
    max_charging_speed: float

@router.post("/evs")
async def create_ev(ev_create: EvCreate, db: Session = Depends(get_db)):
    db_ev = Ev()
    db_ev.name = ev_create.name
    db_ev.battery_capacity = ev_create.battery_capacity
    db_ev.battery_level = ev_create.battery_level
    db_ev.max_charging_speed = ev_create.max_charging_speed
    db.add(db_ev)
    db.commit()
    db.refresh(db_ev)
    return db_ev

@router.get("/evs")
async def get_evs(db: Session = Depends(get_db)):
    return db.query(Ev).all()