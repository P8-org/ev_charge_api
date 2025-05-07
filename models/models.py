import datetime
from sqlalchemy import Column, DateTime, Integer, String, Float, ForeignKey, Enum, Boolean
from sqlalchemy.orm import Session, Mapped, relationship
import enum
from typing import List
from database.base import Base


class State(enum.Enum):
    IDLE = "idle"
    CHARGING = "charging"
    DISCONNECTED = "disconnected"

class Constraint(Base):
    __tablename__ = "constraints"
    id = Column(Integer, primary_key=True, index=True)
    charged_by = Column(DateTime, nullable=False)
    start_time = Column(DateTime, nullable=False)
    target_percentage = Column(Float, nullable=False)
    ev_id = Column(Integer, ForeignKey("user_evs.id"), nullable=False)
    ev: Mapped["UserEV"] = relationship(back_populates="constraints")


class Schedule(Base):
    def __init__(self):
        self.end = datetime.datetime.now()
        self.start = datetime.datetime.now()
        self.schedule_data = ""
        self.num_hours = 0
        self.start_charge = 0
        self.price = 0
        self.greedy_price = 0


    __tablename__ = "schedules"
    id = Column(Integer, primary_key=True, index=True)
    start = Column(DateTime, nullable=False)
    end = Column(DateTime, nullable=False)
    num_hours = Column(Integer, nullable=False)
    start_charge = Column(Float, nullable=False)
    schedule_data = Column(String)  # json format, example: "[20, 20, 0, 0, 0, 10]"
    price = Column(Float, nullable=False)
    greedy_price = Column(Float, nullable=False)
    feasible = Column(Boolean, nullable=False, default=True)
    ev_id = Column(Integer, ForeignKey("user_evs.id"), nullable=False)
    ev: Mapped["UserEV"] = relationship(back_populates="schedule")


class CarModel(Base):
    def __init__(self, model_name: str, model_year: int, battery_capacity: float, max_charging_power: float):
        self.model_name = model_name
        self.model_year = model_year
        self.battery_capacity = battery_capacity
        self.max_charging_power = max_charging_power
        
    __tablename__ = "car_models"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(64), nullable=False)
    model_year = Column(Integer, nullable=False)
    battery_capacity = Column(Float, nullable=False)
    max_charging_power = Column(Float, nullable=False)
    user_evs: Mapped[list["UserEV"]] = relationship("UserEV", back_populates="car_model")


class UserEV(Base):
    __tablename__ = "user_evs"
    id = Column(Integer, primary_key=True, index=True)
    user_set_name = Column(String, nullable=False)
    current_charge = Column(Float, nullable=False)
    current_charging_power = Column(Float, nullable=False)
    max_charging_power = Column(Float, nullable=False)
    state = Column(Enum(State), nullable=False, default="idle")

    # TODO måske slet måske ik
    # active_schedule_id = Column(Integer, ForeignKey("schedules.id"), nullable=True)
    # active_schedule: Mapped["Schedule"] = relationship("Schedule")

    car_model_id = Column(Integer, ForeignKey("car_models.id"), nullable=False)
    car_model: Mapped["CarModel"] = relationship("CarModel", back_populates="user_evs")
    constraints: Mapped[List["Constraint"]] = relationship(
    back_populates="ev", cascade="all, delete-orphan", lazy="joined"
    )
    schedule: Mapped["Schedule"] = relationship(
        back_populates="ev", cascade="all, delete-orphan", uselist=False
    )
