from sqlalchemy import create_engine

from database.base import Base
from sqlalchemy.orm import sessionmaker

from models.models import CarModel

DB_URL = "sqlite:///./database.db" # skal måske være env variable


engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def seed_db():
    db = next(get_db())
    if db.query(CarModel).count() == 0:
        models = [
            CarModel(model_name="Tesla Model Y", model_year=2024, battery_capacity=75.0, max_charging_power=250.0),
            CarModel(model_name="Tesla Model 3", model_year=2024, battery_capacity=82.0, max_charging_power=250.0),
            CarModel(model_name="Volvo EX30", model_year=2024, battery_capacity=69.0, max_charging_power=153.0),
            CarModel(model_name="Skoda Enyaq", model_year=2024, battery_capacity=77.0, max_charging_power=135.0),
            CarModel(model_name="Volkswagen ID.4", model_year=2024, battery_capacity=77.0, max_charging_power=135.0),
            CarModel(model_name="Volkswagen ID.3", model_year=2024, battery_capacity=77.0, max_charging_power=135.0),
            CarModel(model_name="BMW iX1", model_year=2024, battery_capacity=66.5, max_charging_power=130.0),
            CarModel(model_name="MG4", model_year=2024, battery_capacity=64.0, max_charging_power=135.0),
            CarModel(model_name="Audi Q4 e-tron", model_year=2024, battery_capacity=77.0, max_charging_power=135.0),
            CarModel(model_name="BMW i4", model_year=2024, battery_capacity=83.9, max_charging_power=205.0),
            CarModel(model_name="Mercedes-Benz EQA", model_year=2024, battery_capacity=66.5, max_charging_power=100.0),
            CarModel(model_name="Cupra Born", model_year=2024, battery_capacity=77.0, max_charging_power=135.0),
            CarModel(model_name="Volvo EX40", model_year=2024, battery_capacity=78.0, max_charging_power=150.0),
            CarModel(model_name="Hyundai Kona Electric", model_year=2024, battery_capacity=64.8, max_charging_power=100.0),
            CarModel(model_name="Peugeot e-208", model_year=2024, battery_capacity=50.0, max_charging_power=100.0),
            CarModel(model_name="Mercedes-Benz EQB", model_year=2024, battery_capacity=66.5, max_charging_power=100.0),
            CarModel(model_name="Volkswagen ID.7", model_year=2024, battery_capacity=77.0, max_charging_power=170.0),
            CarModel(model_name="Renault Megane E-Tech", model_year=2024, battery_capacity=60.0, max_charging_power=130.0),
            CarModel(model_name="Fiat 500e", model_year=2024, battery_capacity=42.0, max_charging_power=85.0),
            CarModel(model_name="Kia Niro EV", model_year=2024, battery_capacity=64.8, max_charging_power=100.0),
        ]
        db.add_all(models)
        db.commit()