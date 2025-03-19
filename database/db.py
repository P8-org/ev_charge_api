from sqlalchemy import create_engine

from database.base import Base
from sqlalchemy.orm import sessionmaker

DB_URL = "sqlite:///./database.db" # skal måske være env variable


engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()