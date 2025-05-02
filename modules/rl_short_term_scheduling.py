import rl_scheduling


def generate_schedule(num_hours: int, initial_soc: float, battery_capacity: float, max_chargin_rate: float, prices: list[float], print_debug = False) -> list[float]:
    """
    Generates an optimized charging schedule for a battery using reinforcement learning (q-learning).
    Args:
        num_hours (int): The number of hours for which the schedule is to be generated.
        initial_soc (float): The initial state of charge (SOC) of the battery (kWh).
        battery_capacity (float): The total capacity of the battery in kilowatt-hours (kWh).
        max_chargin_rate (float): The maximum charging rate of the battery in kilowatts (kW).
        prices (list[float]): A list of electricity prices for each hour.
        print_debug (bool, optional): If True, debug information will be printed during the training process. Defaults to False.
    Returns:
        list[float]: A list representing the charging schedule, where each element corresponds to the charging rate (in kW) for the respective hour.
    """

    alpha = 0.2 # learning rate
    epsilon = 0.1 # exploration rate
    episodes = 100_000 # episodes

    if (num_hours >= len(prices)): num_hours = len(prices)

    return rl_scheduling.get_schedule(num_hours, alpha, epsilon, episodes, initial_soc, battery_capacity, max_chargin_rate, prices, print_debug)
