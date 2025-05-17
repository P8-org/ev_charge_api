import numpy as np
import cvxpy as cp
import pandas as pd
import requests as req
import datetime as datetime
def optimize_charging_schedule_unused(prices, battery_capacity, initial_soc, max_charging_power, charging_efficiency, deadline):
    """
    POC for solo optimization of schedule without RL.
    
    :param prices: List of electricity prices per time slot (in $/kWh)
    :param battery_capacity: Battery capacity (in kWh)
    :param initial_soc: Initial state of charge (SoC) as a percentage (0-100)
    :param max_charging_power: Maximum charging power (in kW)
    :param charging_efficiency: Charging efficiency (between 0 and 1)
    :param deadline: Number of time slots available for charging (e.g., hours left before needed)
    :return: Optimized charging schedule (list of kWh charged per time slot)

    TODO: 
        make wrapper to receive params from system. 
        use decimal units instead of scientific units
   """
    
    num_slots = min(len(prices), deadline) 
    # Convert initial SoC to kWh
    initial_energy = (initial_soc / 100) * battery_capacity
    required_energy = battery_capacity - initial_energy
    
    # Define decision variable (energy charged per time slot)
    charge = cp.Variable(num_slots, nonneg=True)
    
    # Objective function: Minimize total charging cost
    total_cost = cp.sum(cp.multiply(prices[:num_slots], charge))
    objective = cp.Minimize(total_cost)
    
    # Constraints
    constraints = [
        cp.sum(charge) * charging_efficiency >= required_energy,  # Ensure enough energy is charged
        charge <= max_charging_power  # Limit per-slot charging power
    ]
    
    # Solve optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    print(charge.value)
    
    # Return optimized charging schedule
    return charge.value if charge.value is not None else np.zeros(num_slots)


def adjust_rl_schedule(rl_action, E_required, P_max, first_hour_minutes: int = 0, last_hour_minutes: int = 0):
    """
    Adjusts the RL charging schedule using CVXPY to ensure feasibility.

    Args:
        rl_action (list or np.ndarray): Charging rates from RL agent (kW).
        E_required (float): Required total energy (kWh).
        P_max (float): Max allowable charging rate (kW).

    Returns:
        np.ndarray: Feasible schedule.
    """

    first_hour_mult = (60 - first_hour_minutes) / 60
    last_hour_mult = 1 if last_hour_minutes == 0 else last_hour_minutes / 60

    rl_action = np.array(rl_action)
    n = len(rl_action)
    x = cp.Variable(n)

    # Build multipliers for each hour
    multipliers = np.ones(n)
    if n > 0:
        multipliers[0] = first_hour_mult
    if n > 1:
        multipliers[-1] = last_hour_mult

    objective = cp.Minimize(cp.sum_squares(x - rl_action))

    constraints = [
        x >= 0,
        x <= P_max * multipliers,
        cp.sum(x) >= E_required
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status in ["infeasible","infeasible_inaccurate"]:
        #TODO: implement infeasible problem handling
        print("solution not feasible")
        return rl_action;


    return x.value
