from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.db import get_db


router = APIRouter()

@router.post("/constraints")
async def post_constraint(ev_id: int, db: Session = Depends(get_db)):
    pass