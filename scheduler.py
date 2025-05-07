from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from database.db import get_db
from models.models import UserEV
from routers.schedules import make_schedule

import asyncio

async def update_schedules():
    db = next(get_db())
    evs: list[UserEV] = db.query(UserEV).all()
    for ev in evs:
        try:
            await make_schedule(ev.id, db)
            print("new schedule for ev id", ev.id)
        except: # if schedule is in the past, make_schedule throws an error
            pass 

async def pass_constraints():
    print("Pass constraints")

scheduler = BackgroundScheduler() #Import this to constraints.py and add correct job
update_schedule_trigger = CronTrigger(hour=13, minute=0)
pass_constraint_trigger = CronTrigger(hour=0, minute=0) #implement trigger for constraint start time
scheduler.add_job(lambda: asyncio.run(update_schedule()), trigger=update_schedule_trigger)
scheduler.add_job(lambda: asyncio.run(pass_constraints()), trigger=pass_constraints_trigger)
scheduler.start()
