from fastapi import APIRouter, Depends
from requests import Session
from sqlalchemy import Column, Integer, String

from database.base import Base
from database.db import get_db


router = APIRouter()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)

@router.post("/users")
async def create_user(name: str, db: Session = Depends(get_db)):
    db_user = User()
    db_user.name = name
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.get("/users")
async def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@router.delete("/users/{user_id}")
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    db.delete(db_user)
    db.commit()