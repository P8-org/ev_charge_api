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

from apis.EnergiData import EnergiData, RequestDetail
 
import gymnasium as gym
from gymnasium import spaces

with open("charging_curve.json") as f:
    charging_curves = json.load(f)

class ElectricChargeEnv(gym.Env):
    """
    Custom environment for optimizing electric car charging schedules.
    """
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, prices, times, cars, num_chargers, charge_speed):
        super().__init__()

        self.prices = np.array(prices, dtype=np.float32)
        self.times = np.array(times)
        self.base_cars = cars  # initial cars
        self.num_chargers = num_chargers
        self.charge_speed = charge_speed  # kW
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

    def step(self, action):
        # Limit action to number of free chargers and uncharged cars
        available_chargers = [ch for ch in self.chargers if not ch['in_use']]
        valid_action = min(action, len(available_chargers), len(self.uncharged_car_ids))

        # Assign new cars to free chargers
        for i in range(valid_action):
            charger = available_chargers[i]
            car_id = self.uncharged_car_ids.pop(0)

            charger['in_use'] = True
            charger['car_id'] = car_id
            car = self.cars[car_id]
            # car['charge_start_time'] = self.t
            car['using_charger_id'] = charger['id']
            car['started_at'] = self.t  # record when car starts charging
            car['charge_time'] = self.t # car charging time charging
            self.charging_car_ids.append(car_id)

        # Update charging cars
        cars_fully_charged = []
        for car_id in self.charging_car_ids[:]:  # Copy, might modify list
            car = self.cars[car_id]
            car['charge'] += self.charge_speed
            car['charge_time'] = self.t + 1
            car['charge_percentage'] = (car['charge'] / car['max_charge']) * 100

            if car['charge_percentage'] >= 100:
                car['charged'] = True
                car['charge'] = car['max_charge']
                car['charge_percentage'] = 100

                cars_fully_charged.append(car_id)
                charger_idx = car['using_charger_id']
                self.chargers[charger_idx]['in_use'] = False
                self.chargers[charger_idx]['car_id'] = None
                self.charging_car_ids.remove(car_id)

        # Record finished cars in schedule
        for car_id in cars_fully_charged:
            car = self.cars[car_id]
            # print(car)
            start = car.get('started_at')
            end = self.t + 1
            duration = end - start
            self.schedule.append((start, [car_id], duration))

        # Calculate reward
        cost = self.prices[self.t] * len(self.charging_car_ids)
        reward = -cost

        # Check done
        if self.t + 1 >= self.total_time:
            self.done = True
            if self.uncharged_car_ids:
                reward -= 10 * len(self.uncharged_car_ids)  # Penalty for uncharged cars
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
def run():
    """Runs the training and testing process for the electric charging environment."""
    cars = [
        {'id': 0, 'charged': False, 'charge': 0, 'charge_percentage': 0, 'max_charge': 80, 'using_charger_id': -1},
        {'id': 1, 'charged': False, 'charge': 40, 'charge_percentage': (40/60)*100, 'max_charge': 60, 'using_charger_id': -1},
        {'id': 2, 'charged': False, 'charge': 0, 'charge_percentage': 0, 'max_charge': 60, 'using_charger_id': -1},
    ]
    charge_speed = 12
    num_chargers = 1
    num_episodes = 500
    rd = RequestDetail(
        startDate="StartOfYear-P1M",
        # startDate="StartOfDay-P3D",
        # endDate="StartOfDay-P1M",
        endDate="StartOfDay-P5D",
        dataset="Elspotprices",
        # optional="HourDK,SpotPriceDKK",
        filter_json=json.dumps({"PriceArea": ["DK1"]}),
        limit=24*5, # Default=0, to limit set to a minimum of 72 hours
        # offset=24*0
    )
    data = EnergiData().call_api(rd)
    print(f"Days of data: {len(data)/24}")

    prices = [i.SpotPriceDKK / 1000 for i in data][::-1]
    times = [np.datetime64(i.HourDK) for i in data][::-1]

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
    # print(test_periods)


    agent = None

    if not os.path.isfile("dqn_model.pth"):
        print(f"Number of training periods: {len(train_periods)}")
        for i, (prices_48, times_48) in enumerate(train_periods):
            print(f"\n[Training] Period {i+1}/{len(train_periods)} starting at {times_48[0]}")
            env = ElectricChargeEnv(prices_48, times_48, cars, num_chargers, charge_speed)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            if agent is None:
                agent = DQNAgent(state_dim, action_dim, lr=1e-4)

            # print(agent.action_dim)
            train_agent(env, agent, num_episodes=num_episodes)

        # if (isinstance(agent, DQNAgent)):
        # # Save the trained model
        #     agent.save("dqn_model.pth")

    # ------------------------------
    # Testing the Trained Agent
    # ------------------------------
    prices_48, times_48 = test_periods[0]

    if agent is None:
        env = ElectricChargeEnv(prices_48, times_48, cars, num_chargers, charge_speed)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = DQNAgent(state_dim, action_dim)

    # agent.load("dqn_model.pth")

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
    print("Optimal Charging Schedule (per hour): ")
    for hour, car_ids, charge_time in env.schedule:
        if car_ids:
            print(env.cars[car_ids[0]])
            car_list = ", ".join([f"Car {cid}" for cid in car_ids])
            hour_index = min(hour, len(times_48) - 1)  # prevent out of bounds
            start_time = times_48[hour_index]
            start_time_dt = start_time.astype('M8[s]').tolist()
            start_time_str = start_time_dt.strftime("%Y-%m-%d %H:%M")
            print(f"At {start_time_str} (Price: {prices_48[hour_index]:.2f}) -> Charged: {car_list} (Hour {hour}) Charge Time: {charge_time}")
    
    print(env.schedule)
    # print(charging_curves)

