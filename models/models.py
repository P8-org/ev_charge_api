from sqlalchemy import Column, DateTime, Integer, String, Float, ForeignKey
from sqlalchemy.orm import Session, Mapped, relationship

from database.base import Base


# class Ev(Base):
#     __tablename__ = "evs"
#     id = Column(Integer, primary_key=True, index=True)
#     name: str = Column(String)
#     battery_capacity: float = Column(Float)
#     """ Battery capacity (kwH) """
#     battery_level: float = Column(Float)
#     """ Current battery level (kwH) """
#     max_charging_speed: float = Column(Float)
#     """ Max charging speed (kw) """
#     constraint: Mapped["Constraint"] = relationship(
#         back_populates="ev", cascade="all, delete-orphan"
#     )
#     schedule: Mapped["Schedule"] = relationship(
#         back_populates="ev", cascade="all, delete-orphan"
#     )


class Constraint(Base):
    __tablename__ = "constraints"
    id = Column(Integer, primary_key=True, index=True)
    charged_by = Column(DateTime, nullable=False)
    target_percentage = Column(Float, nullable=False)
    ev_id = Column(Integer, ForeignKey("user_evs.id"), nullable=False)
    ev: Mapped["UserEV"] = relationship(back_populates="constraint")


class Schedule(Base):
    __tablename__ = "schedules"
    id = Column(Integer, primary_key=True, index=True)
    start = Column(DateTime, nullable=False)
    end = Column(DateTime, nullable=False)
    schedule_data = Column(String)  # csv format, example: "20, 20, 0, 0, 0, 10" 
    ev_id = Column(Integer, ForeignKey("user_evs.id"), nullable=False)
    ev: Mapped["UserEV"] = relationship(back_populates="schedule")


class CarModel(Base):
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

    car_model_id = Column(Integer, ForeignKey("car_models.id"), nullable=False)
    car_model: Mapped["CarModel"] = relationship("CarModel", back_populates="user_evs")
    constraint: Mapped["Constraint"] = relationship(
        back_populates="ev", cascade="all, delete-orphan", uselist=False
    )
    schedule: Mapped["Schedule"] = relationship(
        back_populates="ev", cascade="all, delete-orphan", uselist=False
    )