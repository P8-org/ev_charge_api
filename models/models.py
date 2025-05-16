import datetime
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Float, ForeignKey, Enum
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
    end_time = Column(DateTime, nullable=False)
    start_time = Column(DateTime, nullable=False)
    target_percentage = Column(Float, nullable=False)
    ev_id = Column(Integer, ForeignKey("user_evs.id"), nullable=False)
    ev: Mapped["UserEV"] = relationship(back_populates="constraints")


class Schedule(Base):
    __tablename__ = "schedules"
    
    id = Column(Integer, primary_key=True, index=True)
    start = Column(DateTime, nullable=False)
    end = Column(DateTime, nullable=False)
    num_hours = Column(Integer, nullable=False)
    start_charge = Column(Float, nullable=False)
    schedule_data = Column(String)  # csv format, example: "20, 20, 0, 0, 0, 10" 
    price = Column(Float, nullable=False)
    greedy_price = Column(Float, nullable=False)
    feasible = Column(Boolean, nullable=False)
    constraint_id = Column(Integer, ForeignKey("constraints.id"), nullable=False)


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


class UserEV(Base):
    __tablename__ = "user_evs"
    id = Column(Integer, primary_key=True, index=True)
    user_set_name = Column(String, nullable=False)
    current_charge = Column(Float, nullable=False)
    current_charging_power = Column(Float, nullable=False)
    max_charging_power = Column(Float, nullable=False)
    state = Column(Enum(State), nullable=False, default="idle")

    car_model_id = Column(Integer, ForeignKey("car_models.id"), nullable=False)
    car_model: Mapped["CarModel"] = relationship("CarModel")
    constraints: Mapped[List["Constraint"]] = relationship(
    back_populates="ev", cascade="all, delete-orphan", lazy="joined"
    )
    schedule_id = Column(Integer, ForeignKey("schedules.id"), nullable=True)
    schedule: Mapped["Schedule"] = relationship("Schedule")


    def get_next_constraint(self) -> Constraint | None:
        now = datetime.datetime.now()
        return min(
            (constraint for constraint in self.constraints if constraint.end_time > now),
            key=lambda c: c.start_time,
            default=None
        )

    def get_previous_constraint(self) -> Constraint | None:
        now = datetime.datetime.now()
        return max(
            (constraint for constraint in self.constraints if constraint.end_time < now),
            key=lambda c: c.end_time,
            default=None
        )

    def should_generate_new_schedule(self) -> bool:
        next_constraint = self.get_next_constraint()
        if not next_constraint:
            return False

        if not self.schedule:
            return True

        if self.schedule.end < datetime.datetime.now():
            return True
        
        if self.schedule.constraint_id != next_constraint.id:
            return True
            
        return False