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

    def __init__(self, prices, times, car):
        super().__init__()

        self.prices = np.array(prices, dtype=np.float32)
        self.times = np.array(times)
        self.car = car  # single car
        self.total_time = len(prices)
        self.max_price = np.max(self.prices)

        # Fixed observation space: last 48 prices + time features
        self.window_size = 48
        self.action_space = spaces.Discrete(2)  # 0 to num_chargers
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(4 + self.window_size,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0
        self.car["charge"] = 0
        self.car["charge_percentage"] = 0
        self.car["started_at"] = None
        self.car["charge_kw"] = []  # To track charge history (0 or charge_speed)
        self.done = False
        self.schedule = []  # (time, charge amount)

        return self._get_state(), {}

    def _get_state(self):
        # Use the last `window_size` prices
        start_idx = max(0, self.t - self.window_size + 1)
        end_idx = self.t + 1
        recent_prices = self.prices[start_idx:end_idx]
        padded_prices = np.zeros(self.window_size, dtype=np.float32)
        padded_prices[-len(recent_prices):] = recent_prices / (self.max_price + 1e-6)

        norm_time = self.t / (self.total_time - 1)
        norm_remaining = (self.car["max_charge_kw"] - self.car["charge"]) / self.car["max_charge_kw"]
        current_time = self.times[self.t].astype('datetime64[s]').astype(object)
        hour_of_day = current_time.hour / 23.0
        day_of_week = current_time.weekday() / 6.0

        return np.concatenate(([norm_time, norm_remaining, hour_of_day, day_of_week], padded_prices)).astype(np.float32)

    def _should_charge_now(self, car, hours_left):
        """
        Decide whether to charge the car now based on future prices, constraints, and urgency.
        """
        remaining_charge = car['max_charge_kw'] - car['charge']
        hours_needed = int(np.ceil(remaining_charge / car['charge_speed']))

        constraints = car['constraints']
        start = constraints.get('start', 0)
        end = constraints.get('end', self.total_time)

        # If current time is before the start constraint, do not charge
        if self.t < start:
            return False

        # If current time is after the end constraint, do not charge
        if self.t > end:
            return False

        # Urgency: Charge now if not enough hours are left to meet the charge goal
        # if hours_needed >= (end - self.t):
        #     return True

        # Consider future prices within the allowed window
        future_prices = self.prices[self.t:end]

        if not future_prices.size:
            return False  # No future prices to consider

        # Heuristic: Charge if current price is within the cheapest N hours
        N = min(hours_needed, len(future_prices))  # Number of hours we need to charge
        cheapest_hours = np.partition(future_prices, N-1)[:N]  # Top N cheapest prices
        current_price = self.prices[self.t]

        return current_price <= max(cheapest_hours)

    def step(self, action):
        # Charge the car if action is 1
        reward = 0
        hours_left = self.total_time - self.t
        end_constraint = self.car['constraints'].get('end', self.total_time)
        time_until_end = end_constraint - self.t

        # Encourage early and steady charging
        if self.car["charge_percentage"] < 100 and time_until_end <= 1:
            reward -= 20  # Penalize almost-out-of-time undercharging
        elif action == 1 and self.car["charge_percentage"] >= 100:
            reward -= 5   # Penalize overcharging
        elif action == 1:
            reward += 1   # Small reward for charging

        if self.car["charge_percentage"] < 100 and self._should_charge_now(self.car, hours_left):
            if self.car["started_at"] is None:
                self.car["started_at"] = self.t

            self.car["charge"] += self.car["charge_speed"]
            self.car["charge_percentage"] = min(
                (self.car["charge"] / self.car["max_charge_kw"]) * 100, 100
            )
            self.car["charge_kw"].append(self.car["charge_speed"])  # Log charging for this hour
        else:
            # Not charging this hour, fill in the gap if charging has started
            if self.car["started_at"] is not None and time_until_end > 0:
                self.car["charge_kw"].append(0)


        # Calculate cost and reward
        cost = self.prices[self.t] * self.car["charge_speed"] if action == 1 else 0
        price_sensitivity = 1.0
        reward = -cost * price_sensitivity  # minimize cost

        # Check end condition
        if self.t + 1 >= self.total_time:
            self.done = True

            if self.car["charge_percentage"] < 100:
                reward -= 100 * (100 - self.car["charge_percentage"])
        elif self.car["charge_percentage"] >= 100:
            self.done = True

        if self.car["started_at"] is not None and (self.car["charge_percentage"] >= 100 or self.done):
            # Finalize the last charging session if still active
            if self.car["started_at"] is not None:
                self.schedule.append({
                    "start_time": self.car["started_at"],
                    "charge": min(self.car["charge"], self.car['max_charge_kw']),
                    "charge_percentage": self.car["charge_percentage"],
                    "charge_kw": self.car["charge_kw"].copy()
                })
                self.car["started_at"] = None
                self.car["charge_kw"] = []

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

def train_dqn(cars: list[dict], rd:RequestDetail, num_chargers:int = None, num_episodes=2000, period_length=48):
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
    p_size = period_length 
    for start_idx in range(0, len(prices_np) - p_size, p_size):
        prices_48 = prices_np[start_idx:start_idx + p_size]
        times_48 = times_np[start_idx:start_idx + p_size]
        periods.append((prices_48, times_48))

    train_periods = periods

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
            env = ElectricChargeEnv(prices_48, times_48, cars[0])
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
def run_dqn(car: dict, rd:RequestDetail):
    data = EnergiData().call_api(rd)
    print(f"Days of data: {len(data)/24}")

    prices = [i.SpotPriceDKK / 1000 for i in data]
    times = [np.datetime64(i.HourDK) for i in data]

    prices_np = np.asarray(prices, dtype=np.float32)
    times_np = np.asarray(times, dtype=np.datetime64)

    prices_48 = prices_np[0:]
    times_48 = times_np[0:]


    env = ElectricChargeEnv(prices_48, times_48, car)
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

    # print(env.schedule)    
    # print(prices_48)
    # prices_48.sort()
    # print(prices_48[:4])
    return env.schedule
