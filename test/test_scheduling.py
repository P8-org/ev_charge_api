import datetime
from models.models import CarModel, Constraint, Schedule, UserEV
from modules.rl_short_term_scheduling import generate_schedule


def test_partial_hours_1():
    prices = [0, 0, 0, 0]
    num_hours = 4
    target_kwh = 100
    max_power = 10
    start_time_minute = 15
    end_time_minute = 15
    schedule_data = generate_schedule(num_hours, 0, target_kwh, max_power, prices, start_time_minute, end_time_minute, False)

    assert schedule_data == [7.5, 10, 10, 2.5]

def test_partial_hours_2():
    prices = [0, 0, 0, 0]
    num_hours = 4
    target_kwh = 100
    max_power = 10
    start_time_minute = 45
    end_time_minute = 45
    schedule_data = generate_schedule(num_hours, 0, target_kwh, max_power, prices, start_time_minute, end_time_minute, False)

    assert schedule_data == [2.5, 10, 10, 7.5]

def test_partial_hours_3():
    prices = [1, 1]
    num_hours = 2
    target_kwh = 100
    max_power = 10
    start_time_minute = 45
    end_time_minute = 45
    schedule_data = generate_schedule(num_hours, 0, target_kwh, max_power, prices, start_time_minute, end_time_minute, False)

    assert schedule_data == [2.5, 7.5]


def test_partial_hours_4():
    prices = [0]
    num_hours = 1
    target_kwh = 100
    max_power = 10
    start_time_minute = 30
    end_time_minute = 45
    schedule_data = generate_schedule(num_hours, 0, target_kwh, max_power, prices, start_time_minute, end_time_minute, False)

    assert schedule_data == [2.5]


def test_should_generate_1():
    ev = UserEV()
    constraint = Constraint()
    constraint.start_time = datetime.datetime.now()
    constraint.end_time = datetime.datetime.now() + datetime.timedelta(hours=1)
    ev.constraints = [constraint]
    ev.schedule = None

    assert ev.should_generate_new_schedule() == True

def test_should_generate_2():
    ev = UserEV()
    constraint1 = Constraint()
    constraint1.start_time = datetime.datetime.now() - datetime.timedelta(hours=2)
    constraint1.end_time = datetime.datetime.now() - datetime.timedelta(hours=1)
    constraint1.id = 1

    constraint2 = Constraint()
    constraint2.start_time = datetime.datetime.now() + datetime.timedelta(hours=2)
    constraint2.end_time = datetime.datetime.now() + datetime.timedelta(hours=3)
    constraint2.id = 2
    
    ev.constraints = [constraint1, constraint2]
    schedule = Schedule()
    schedule.constraint_id = 1
    schedule.start = constraint1.start_time
    schedule.end = constraint1.end_time
    ev.schedule = schedule

    assert ev.should_generate_new_schedule() == True


def test_should_not_generate_1():
    ev = UserEV()
    ev.constraints = []
    ev.schedule = None

    assert ev.should_generate_new_schedule() == False


def test_should_not_generate_2():
    ev = UserEV()
    constraint1 = Constraint()
    constraint1.start_time = datetime.datetime.now() - datetime.timedelta(hours=1)
    constraint1.end_time = datetime.datetime.now() + datetime.timedelta(hours=1)
    constraint1.id = 1

    constraint2 = Constraint()
    constraint2.start_time = datetime.datetime.now() + datetime.timedelta(hours=2)
    constraint2.end_time = datetime.datetime.now() + datetime.timedelta(hours=3)
    constraint2.id = 2
    
    ev.constraints = [constraint1, constraint2]
    schedule = Schedule()
    schedule.constraint_id = 1
    schedule.start = constraint1.start_time
    schedule.end = constraint1.end_time
    ev.schedule = schedule

    assert ev.should_generate_new_schedule() == False

def test_schedule_with_zero_target_kwh():
    prices = [1, 2, 3, 4]
    num_hours = 4
    target_kwh = 0
    max_power = 10
    start_time_minute = 0
    end_time_minute = 0
    schedule_data = generate_schedule(num_hours, 0, target_kwh, max_power, prices, start_time_minute, end_time_minute, False)
    assert schedule_data == [0.0, 0.0, 0.0, 0.0]

def test_schedule_with_partial_first_and_last_hour():
    prices = [1, 2, 3]
    num_hours = 3
    target_kwh = 15
    max_power = 10
    start_time_minute = 30
    end_time_minute = 30
    schedule_data = generate_schedule(num_hours, 0, target_kwh, max_power, prices, start_time_minute, end_time_minute, False)
    # 0.5h * 10 = 5, 1h * 10 = 10, 0.5h * 10 = 5, but only 15kWh needed, so [5, 10, 0]
    assert sum(schedule_data) == 15
    assert schedule_data[0] == 5.0
    assert schedule_data[1] == 10.0
    assert schedule_data[2] == 0.0

def test_schedule_with_prices_affecting_distribution():
    prices = [10, 1, 5, 2]
    num_hours = 4
    target_kwh = 20
    max_power = 10
    start_time_minute = 0
    end_time_minute = 0
    schedule_data = generate_schedule(num_hours, 0, target_kwh, max_power, prices, start_time_minute, end_time_minute, True)
    assert sum(schedule_data) == 20
    assert schedule_data[1] == 10.0  # hour with price 1
    assert schedule_data[3] == 10.0  # hour with price 2

def test_schedule_with_insufficient_time():
    prices = [1, 1]
    num_hours = 2
    target_kwh = 30
    max_power = 10
    start_time_minute = 0
    end_time_minute = 0
    schedule_data = generate_schedule(num_hours, 0, target_kwh, max_power, prices, start_time_minute, end_time_minute, False)
    # Only 20kWh can be delivered, so schedule should max out both hours
    assert schedule_data == [10.0, 10.0]

def test_schedule_with_exact_partial_hours():
    prices = [1, 1]
    num_hours = 2
    target_kwh = 10
    max_power = 10
    start_time_minute = 30
    end_time_minute = 30
    schedule_data = generate_schedule(num_hours, 0, target_kwh, max_power, prices, start_time_minute, end_time_minute, False)
    # 0.5h + 0.5h = 1h, so 10kWh in total, 5kWh each
    assert schedule_data == [5.0, 5.0]
    assert sum(schedule_data) == 10


def test_schedule_with_half_charged():
    prices = [1, 1, 2, 1]
    num_hours = 4
    target_kwh = 50
    max_power = 10
    start_time_minute = 0
    end_time_minute = 0
    schedule_data = generate_schedule(num_hours, 25, target_kwh, max_power, prices, start_time_minute, end_time_minute, False)
    assert sum(schedule_data) == 25
    assert schedule_data[2] == 0


def test_full_hours():
    prices = [1, 1, 2, 1, 1]
    num_hours = 5
    target_kwh = 50
    max_power = 15
    start_time_minute = 0
    end_time_minute = 0
    schedule_data = generate_schedule(num_hours, 0, target_kwh, max_power, prices, start_time_minute, end_time_minute, False)
    assert sum(schedule_data) == 50
    assert schedule_data[2] == 0


def test_full_hours_2():
    prices = [1, 1]
    num_hours = 2
    target_kwh = 50
    max_power = 25
    start_time_minute = 0
    end_time_minute = 0
    schedule_data = generate_schedule(num_hours, 0, target_kwh, max_power, prices, start_time_minute, end_time_minute, False)
    assert schedule_data == [25, 25]


def test_full_hours_3():
    prices = [1]
    num_hours = 1
    target_kwh = 50
    max_power = 25
    start_time_minute = 0
    end_time_minute = 0
    schedule_data = generate_schedule(num_hours, 0, target_kwh, max_power, prices, start_time_minute, end_time_minute, False)
    assert schedule_data == [25]