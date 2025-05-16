import datetime
from models.models import UserEV, Constraint, Schedule, CarModel
from routers.evs import simulate_charging

ev = UserEV()
ev.max_charging_power = 10
ev.current_charge = 0
ev.current_charging_power = 0
ev.car_model = CarModel(model_name="Name", model_year=2025, battery_capacity=100, max_charging_power=10)

def test_sim_charging_no_schedule():
    simulate_charging(ev=ev)

    assert ev.current_charge == 0
    assert ev.current_charging_power == 0


def test_sim_charging_after_schedule_end():
    schedule = Schedule()
    schedule.num_hours = 2
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 15, 23)
    schedule.end = datetime.datetime(2025, 5, 16, 1)
    schedule.schedule_data = "10, 5"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 3))

    assert ev.current_charge == 15
    assert ev.current_charging_power == 0




def test_sim_charging_after_end_2():
    schedule = Schedule()
    schedule.num_hours = 6
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 15, 23)
    schedule.end = datetime.datetime(2025, 5, 16, 5)
    schedule.schedule_data = "10.0, 10.0, 10.0, 10.0, 0.0, 4.5"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 7))

    assert ev.current_charge == 44.5
    assert ev.current_charging_power == 0



def test_sim_charging_before_start():
    schedule = Schedule()
    schedule.num_hours = 8
    schedule.start_charge = 0
    schedule.start = datetime.datetime.now() + datetime.timedelta(minutes=1)
    schedule.end = datetime.datetime.now() + datetime.timedelta(hours=8, minutes=1)
    schedule.schedule_data = "10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0"
    
    ev.schedule = schedule

    simulate_charging(ev=ev)

    assert ev.current_charge == 0
    assert ev.current_charging_power == 0



def test_sim_charging_1():
    schedule = Schedule()
    schedule.num_hours = 8
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 6)
    schedule.end = datetime.datetime(2025, 5, 16, 14)
    schedule.schedule_data = "10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 10))

    assert ev.current_charge == 40
    assert ev.current_charging_power != 0


def test_sim_charging_2():
    schedule = Schedule()
    schedule.num_hours = 8
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 6)
    schedule.end = datetime.datetime(2025, 5, 16, 14)
    schedule.schedule_data = "2, 4, 6, 8, 10, 12, 14, 16"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 9, 30))

    assert ev.current_charge == 16
    assert ev.current_charging_power != 0



def test_sim_charging_3():
    schedule = Schedule()
    schedule.num_hours = 3
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 9, 30)
    schedule.end = datetime.datetime(2025, 5, 16, 11, 30)
    schedule.schedule_data = "5.0, 10.0, 5.0"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 10, 0))

    assert ev.current_charge == 5
    assert ev.current_charging_power != 0

def test_sim_charging_4():
    schedule = Schedule()
    schedule.num_hours = 1
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 10, 0)
    schedule.end = datetime.datetime(2025, 5, 16, 10, 30)
    schedule.schedule_data = "5.0"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 11, 30))

    assert ev.current_charge == 5
    assert ev.current_charging_power == 0


def test_sim_charging_5():
    schedule = Schedule()
    schedule.num_hours = 1
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 10, 0)
    schedule.end = datetime.datetime(2025, 5, 16, 10, 30)
    schedule.schedule_data = "5.0"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 10, 15))

    assert ev.current_charge == 2.5
    assert ev.current_charging_power != 0

def test_sim_charging_6():
    schedule = Schedule()
    schedule.num_hours = 1
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 10, 15)
    schedule.end = datetime.datetime(2025, 5, 16, 10, 45)
    schedule.schedule_data = "5.0"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 10, 45))

    assert ev.current_charge == 5
    assert ev.current_charging_power == 0


def test_sim_charging_7():
    schedule = Schedule()
    schedule.num_hours = 1
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 10, 45)
    schedule.end = datetime.datetime(2025, 5, 16, 11, 0)
    schedule.schedule_data = "2.5"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 10, 50))

    assert round(ev.current_charge, 2) == 0.83
    assert ev.current_charging_power != 0


def test_sim_charging_8():
    schedule = Schedule()
    schedule.num_hours = 2
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 10, 0)
    schedule.end = datetime.datetime(2025, 5, 16, 12, 0)
    schedule.schedule_data = "10, 12"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 11, 30))

    assert ev.current_charge == 16
    assert ev.current_charging_power != 0


def test_sim_charging_9():
    schedule = Schedule()
    schedule.num_hours = 2
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 10, 0)
    schedule.end = datetime.datetime(2025, 5, 16, 12, 0)
    schedule.schedule_data = "10, 0"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 11, 30))

    assert ev.current_charge == 10
    assert ev.current_charging_power == 0


def test_sim_charging_10():
    schedule = Schedule()
    schedule.num_hours = 1
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 10, 30)
    schedule.end = datetime.datetime(2025, 5, 16, 10, 45)
    schedule.schedule_data = "0"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 10, 35))

    assert ev.current_charge == 0
    assert ev.current_charging_power == 0

def test_sim_charging_11():
    schedule = Schedule()
    schedule.num_hours = 3
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 9, 30)
    schedule.end = datetime.datetime(2025, 5, 16, 11, 30)
    schedule.schedule_data = "5.0, 10.0, 5.0"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 9, 45))

    assert ev.current_charge == 2.5
    assert ev.current_charging_power != 0

def test_sim_charging_12():
    schedule = Schedule()
    schedule.num_hours = 3
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 9, 30)
    schedule.end = datetime.datetime(2025, 5, 16, 11, 30)
    schedule.schedule_data = "5.0, 10.0, 5.0"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 11, 0))

    assert ev.current_charge == 15
    assert ev.current_charging_power != 0


def test_sim_charging_13():
    schedule = Schedule()
    schedule.num_hours = 3
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 9, 30)
    schedule.end = datetime.datetime(2025, 5, 16, 11, 30)
    schedule.schedule_data = "5.0, 10.0, 5.0"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 11, 15))

    assert ev.current_charge == 17.5
    assert ev.current_charging_power != 0


def test_sim_charging_at_end():
    schedule = Schedule()
    schedule.num_hours = 3
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 9, 30)
    schedule.end = datetime.datetime(2025, 5, 16, 11, 30)
    schedule.schedule_data = "5.0, 10.0, 5.0"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 11, 30))

    assert ev.current_charge == 20
    assert ev.current_charging_power == 0


def test_sim_charging_at_beginning():
    schedule = Schedule()
    schedule.num_hours = 3
    schedule.start_charge = 0
    schedule.start = datetime.datetime(2025, 5, 16, 9, 30)
    schedule.end = datetime.datetime(2025, 5, 16, 11, 30)
    schedule.schedule_data = "5.0, 10.0, 5.0"
    
    ev.schedule = schedule

    simulate_charging(ev=ev, now=datetime.datetime(2025, 5, 16, 9, 30))

    assert ev.current_charge == 0
    assert ev.current_charging_power != 0

