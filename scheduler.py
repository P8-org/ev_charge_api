from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from database.db import get_db
from models.models import UserEV
from routers.schedules import make_schedule

async def my_func():
    db = next(get_db())
    evs: list[UserEV] = db.query(UserEV).all()
    for ev in evs:
        try:
            await make_schedule(ev.id, db)
            print("new schedule for ev id", ev.id)
        except: # if schedule is in the past, make_schedule throws an error
            pass 

scheduler = BackgroundScheduler()
trigger = CronTrigger(hour=13, minute=0)
scheduler.add_job(my_func, trigger=trigger)
scheduler.start()
