from datetime import datetime, timedelta
from dateutil import parser
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import json
from rich import print
import os.path
from models.models import Schedule

from apis.EnergiData import EnergiData, RequestDetail
 
import gymnasium as gym
from gymnasium import spaces


class ElectricChargeEnv(gym.Env):
    """
    Custom environment for optimizing electric car charging schedules.
    """
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, prices, times, cars, num_chargers):
        super().__init__()

        self.prices = np.array(prices, dtype=np.float32)
        self.times = np.array(times)
        self.base_cars = cars  # initial cars
        self.num_chargers = num_chargers
        self.total_time = len(prices)
        self.max_price = np.max(self.prices)

        self.action_space = spaces.Discrete(num_chargers + 1)  # 0 to num_chargers
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(4 + self.total_time,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0
        self.cars = [car.copy() for car in self.base_cars]  # Deep copy cars
        self.uncharged_car_ids = [car['id'] for car in self.cars]
        self.charging_car_ids = []
        self.chargers = [{'id': i, 'in_use': False, 'car_id': None} for i in range(self.num_chargers)]

        self.done = False
        self.schedule = []  # (time, list of car ids, charge time)

        return self._get_state(), {}

    def _get_state(self):
        norm_time = self.t / (self.total_time - 1)
        norm_remaining = len(self.uncharged_car_ids) / len(self.cars)
        norm_prices = self.prices / (self.max_price + 1e-6)

        current_time = self.times[self.t].astype('datetime64[s]').astype(object)
        hour_of_day = current_time.hour / 23.0
        day_of_week = current_time.weekday() / 6.0

        return np.concatenate(([norm_time, norm_remaining, hour_of_day, day_of_week], norm_prices)).astype(np.float32)

    def _should_charge_now(self, car, hours_left):
        """
        Decide whether to charge the car now based on future prices and urgency.
        """
        remaining_charge = car['max_charge'] - car['charge']
        hours_needed = int(np.ceil(remaining_charge / car['charge_speed']))

        constraints = car['constraints']
        if 'end' in constraints and hours_needed >= (constraints['end'] - self.t):
            return True

        if 'start' in constraints and self.t < constraints['start']:
            return False 

        if 'start' in constraints and 'end' in constraints:
            start = constraints['start']
            end = constraints['end']
            if not (start <= self.t < end):
                return False

        if 'end' in constraints:
            future_prices = self.prices[self.t:constraints['end']]
            hours_to_consider = min(hours_left, constraints['end'] - self.t)
        else:
            future_prices = self.prices[self.t:self.t + hours_left]
            hours_to_consider = hours_left

        if not future_prices.size:
            return False # No future prices to consider

        current_price = self.prices[self.t]

        # Heuristic: only charge now if price is within lowest N of remaining window
        N = min(hours_needed, hours_to_consider)  # Number of hours we need to charge
        cheapest_hours = sorted(future_prices)[:N]

        if not cheapest_hours:
            return False

        return current_price <= max(cheapest_hours)

    def step(self, action):
        # Determine number of cars to charge this step (based on action and available chargers)
        total_available = min(action, len(self.chargers))
        hours_left = self.total_time - self.t

        candidates = [car_id for car_id in self.uncharged_car_ids + self.charging_car_ids
                      if self.cars[car_id]['charge_percentage'] < 100]

        # Filter candidates
        eligible_cars = []
        for car_id in candidates:
            car = self.cars[car_id]
            if self._should_charge_now(car, hours_left):
                eligible_cars.append(car_id)

        
        cars_to_charge = eligible_cars[:total_available]
        self.charging_car_ids = []

        active_car_ids = [cid for cid in self.uncharged_car_ids + self.charging_car_ids if self.cars[cid]['charge_percentage'] < 100]

        for car_id in active_car_ids:
            car = self.cars[car_id]


            if car_id in cars_to_charge:
                if 'started_at' not in car:
                    car['started_at'] = self.t
                    car['charge_kw'] = []  # Start log when charging process begins

                # Charge
                car['charge'] += car['charge_speed']
                car['charge_time'] = self.t + 1
                car['charge_percentage'] = (car['charge'] / car['max_charge']) * 100
                car['charge_kw'].append(car['charge_speed'])
                self.charging_car_ids.append(car_id)
            else:
                # Not charging this hour
                if 'charge_kw' in car: # Only append 0 if started charging
                    car['charge_kw'].append(0)

            # Finalize if fully charged
            if car['charge_percentage'] >= 100:
                car['charged'] = True
                car['charge'] = car['max_charge']
                car['charge_percentage'] = 100

                if car_id in self.uncharged_car_ids:
                    self.uncharged_car_ids.remove(car_id)

                charge_kw = car['charge_kw']
                start = car['started_at']
                end = start + len(charge_kw)
                duration = len(charge_kw)

                # total_cost = sum(charge_kw[i] * float(self.prices[start + i]) for i in range(duration))

                self.schedule.append({
                    "car_id": car_id,
                    "start_time": start,
                    "end_time": end,
                    "charge_speed": car['charge_speed'],
                    "charge_kw": charge_kw,
                    "duration": duration
                })

        # Base cost for current step
        cost = self.prices[self.t] * sum(
            1 for cid in self.charging_car_ids if self.cars[cid]['charge_kw'][-1] > 0
        )
        price_sensitivity = 10.0
        reward = -cost * price_sensitivity # minimize cost

        # Check end condition
        if self.t + 1 >= self.total_time:
            self.done = True
            for car_id in self.uncharged_car_ids + self.charging_car_ids:
                charge_percentage = self.cars[car_id]['charge_percentage']
                if charge_percentage <= 50: #TODO replace with car's minimum required charge percentage'
                    reward -= 400 * (100 - self.cars[car_id]['charge_percentage'])
                elif charge_percentage < 100:
                    reward -= 100 * (100 - self.cars[car_id]['charge_percentage'])
        elif not self.uncharged_car_ids and not self.charging_car_ids:
            self.done = True

        self.t = min(self.t + 1, self.total_time - 1)

        return self._get_state(), reward, self.done, False, {}

    def render(self, mode="human"):
        print(f"Time Step {self.t}: Schedule -> {self.schedule}")

    def close(self):
        pass

# ------------------------------
# Q-Network Definition
# ------------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ------------------------------
# DQN Agent and Training Setup
# ------------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.action_dim = action_dim

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
        target = rewards + self.gamma * next_q_values * (1 - dones)
        loss = torch.nn.MSELoss()(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"[green]Model saved to {path}[/green]")

    def load(self, path):
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', 1.0)
            print(f"[cyan]Model loaded from {path}[/cyan]")
        else:
            print(f"[red]No model found at {path}[/red]")


# ------------------------------
# Training Loop
# ------------------------------
def train_agent(env, agent, num_episodes=500, print_iter=False):
    update_target_every = 10  # update target network every 10 episodes
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            agent.replay_buffer.append((state, action, reward, next_state, float(done)))
            state = next_state

            agent.update()
        
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)

        if episode % update_target_every == 0:
            agent.update_target()

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    return agent

# ------------------------------
# Usage
# ------------------------------

def save_progress(last_index):
    with open("./train_progress", "w") as f:
        json.dump({"last_trained_index": last_index}, f)

def load_progress():
    if os.path.exists("./train_progress"):
        with open("./train_progress", "r") as f:
            return json.load(f).get("last_trained_index", -1)
    else:
        print("[yellow]The file does not exist, could not be loaded[/yellow]") 
        return -1

def remove_progress():
    if os.path.exists("./train_progress"):
        os.remove("./train_progress")
        os.remove("./dqn_model_temp.pth")
    else:
        print("[yellow]The file does not exist, could not be removed[/yellow]") 

def process_data(cars: list[dict], rd:RequestDetail, num_chargers:int = None, num_episodes=2000):
    """Runs the training and testing process for the electric charging environment."""

    data = EnergiData().call_api(rd)
    print(f"Days of data: {len(data)/24}")

    prices = [i.SpotPriceDKK / 1000 for i in data]
    times = [np.datetime64(i.HourDK) for i in data]

    prices_np = np.asarray(prices, dtype=np.float32)
    times_np = np.asarray(times, dtype=np.datetime64)

    # Create 48-hour periods
    periods = []
    for start_idx in range(0, len(prices_np) - 47, 24):
        prices_48 = prices_np[start_idx:start_idx + 24]
        times_48 = times_np[start_idx:start_idx + 24]
        periods.append((prices_48, times_48))
    
    return periods

def run_dqn(cars: list[dict], rd:RequestDetail, num_chargers:int = None, num_episodes=2000):
    if num_chargers is None:
        num_chargers = len(cars)
    data = EnergiData().call_api(rd)
    print(f"Days of data: {len(data)/24}")

    prices = [i.SpotPriceDKK / 1000 for i in data]
    times = [np.datetime64(i.HourDK) for i in data]

    prices_np = np.asarray(prices, dtype=np.float32)
    times_np = np.asarray(times, dtype=np.datetime64)

    # Create 48-hour periods
    periods = []
    for start_idx in range(0, len(prices_np) - 47, 24):
        prices_48 = prices_np[start_idx:start_idx + 24]
        times_48 = times_np[start_idx:start_idx + 24]
        periods.append((prices_48, times_48))

    # split_idx = int(0.8 * len(periods))
    split_idx = len(periods) - 1 
    train_periods = periods[:split_idx]
    test_periods = periods[split_idx:]

    agent = None
    last_trained_index = load_progress()
    finished = False

    if os.path.isfile("dqn_model.pth"):
        finished = True

    if finished is False and not len(train_periods) == last_trained_index:
        print(f"Number of training periods: {len(train_periods)}")
        for i, (prices_48, times_48) in enumerate(train_periods):
            if i <= last_trained_index:
                continue  # Skip already trained periods

            print(f"\n[Training] Period {i+1}/{len(train_periods)} starting at {times_48[0]}")
            env = ElectricChargeEnv(prices_48, times_48, cars, num_chargers)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            if agent is None:
                agent = DQNAgent(state_dim, action_dim, lr=1e-4)

            # print(agent.action_dim)
            train_agent(env, agent, num_episodes=num_episodes)
            save_progress(i)

            agent.save("dqn_model_temp.pth")

        if (isinstance(agent, DQNAgent)):
        # Save the trained model
            agent.save("dqn_model.pth")

        remove_progress()

    # ------------------------------
    # Testing the Trained Agent
    # ------------------------------
    prices_48, times_48 = test_periods[0]

    env = ElectricChargeEnv(prices_48, times_48, cars, num_chargers)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    agent.load("dqn_model.pth")

    print("\n[bold underline]Testing Trained Agent[/bold underline]")
    print(f"\n[Testing] starting at {times_48[0]}")
    state, _ = env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_values = agent.q_network(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        state, _, done, _, _ = env.step(action)

    # Print the schedule
    # print("Optimal Charging Schedule (per hour): ")
    for car in env.schedule:
        hour_index = min(car['start_time'], len(times_48) - 1)  # prevent out of bounds
        start_time = times_48[hour_index]
        start_time_dt = start_time.astype('M8[s]').tolist()
        start_time_str = start_time_dt.strftime("%Y-%m-%d %H:%M")
        print(f"At {start_time_str} (Price: {prices_48[hour_index]:.2f}) -> Charged: {car['car_id']} (Hour {car['start_time']}) Charge Time: {car['duration']}")

    print(env.schedule)    
