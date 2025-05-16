import datetime
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
            make_schedule(ev.id, db)
            print("new schedule for ev id", ev.id)
        except: # if schedule is in the past, make_schedule throws an error
            pass 



scheduler = BackgroundScheduler()
trigger = CronTrigger(hour=13, minute=0)
scheduler.add_job(lambda: asyncio.run(update_schedules()), trigger=trigger)
scheduler.start()


def schedule_next(ev_id: int, run_time: datetime.datetime):
    scheduler.add_job(make_schedule,
        'date',
        run_date=run_time,
        args=[ev_id, next(get_db())],
        id=f"ev_{ev_id}_schedule",
        replace_existing=True
    )

def reschedule_all():
    db = next(get_db())
    evs = db.query(UserEV).all()
    now = datetime.datetime.now()
    for ev in evs:
        if ev.schedule and ev.schedule.end > now:
            schedule_next(ev.id, ev.schedule.end)

# Call this on startup
reschedule_all()