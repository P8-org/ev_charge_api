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

    def __init__(self, prices, times, cars, num_chargers=1):
        super().__init__()
        self.prices = np.array(prices, dtype=np.float32)
        self.times = np.array(times)
        self.window_size = 48
        self.total_time = len(self.prices)
        self.max_price = np.max(self.prices)
        self.num_chargers = num_chargers
        self.base_cars = cars
        self.num_cars = max(len(cars),1)
        self.cars = []
        self.max_cars = 10

        self._define_spaces(self.max_cars)

        self.reset()

    def _define_spaces(self, num_cars):
        self.action_space = spaces.MultiBinary(num_cars)
        
        obs_dim = 3 + self.window_size + self.max_cars
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def reset(self, num_cars=None, num_chargers=None):
        if num_cars is not None:
            self.num_cars = min(num_cars, self.max_cars)
        if num_chargers is not None:
            self.num_chargers = num_chargers

        self.t = 0
        self.done = False

        # Create deep copies of the cars to avoid modifying the original list
        self.cars = []
        for car in self.base_cars[:self.num_cars]:
            car_copy = car.copy()
            car_copy["started_at"] = None
            car_copy["charge_kw"] = []
            car_copy["charge_percentage"] = min((car_copy["charge"] / car_copy["max_charge_kw"]) * 100, 100)
            self.cars.append(car_copy)

        self.schedule = []

        return self._get_state(), {}

    def _get_state(self):
        # Use the last `window_size` prices
        start_idx = max(0, self.t - self.window_size + 1)
        end_idx = self.t + 1
        recent_prices = self.prices[start_idx:end_idx]
        padded_prices = np.zeros(self.window_size, dtype=np.float32)
        padded_prices[-len(recent_prices):] = recent_prices / (self.max_price + 1e-6)

        norm_time = self.t / (self.total_time - 1)
        
        # Create padded remaining charge for cars (fixed size for max_cars)
        norm_remaining = np.zeros(self.max_cars, dtype=np.float32)
        for i, car in enumerate(self.cars):
            if i < self.max_cars:
                norm_remaining[i] = (car["max_charge_kw"] - car["charge"]) / car["max_charge_kw"]
        
        current_time = self.times[self.t].astype('datetime64[s]').astype(object)
        hour_of_day = current_time.hour / 23.0
        day_of_week = current_time.weekday() / 6.0

        return np.concatenate(([norm_time, hour_of_day, day_of_week], padded_prices, norm_remaining)).astype(np.float32)

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

    def step(self, actions):
        reward = 0
        hours_left = self.total_time - self.t
        end_constraint = [car['constraints'].get('end', self.total_time) for car in self.cars]
        time_until_end = [end - self.t for end in end_constraint]

        # handling for cars with tight constraints
        for i, car in enumerate(self.cars):
            constraints = car['constraints']
            start = constraints.get('start', 0)
            end = constraints.get('end', self.total_time)
            
            if self.t >= start and self.t < end and car["charge_percentage"] < 100:
                remaining_charge = car["max_charge_kw"] - car["charge"]
                hours_needed = np.ceil(remaining_charge / car["charge_speed"])
                time_left = end - self.t
                
                if hours_needed >= time_left and i < len(actions):
                    actions[i] = 1  # Force charge

        # calculate priority for each car
        charging_priorities = []
        for i, car in enumerate(self.cars):
            if i < len(actions):
                urgency = 1.0
                if time_until_end[i] > 0:

                    remaining_percentage = 100 - (car["charge"] / car["max_charge_kw"]) * 100
                    urgency = remaining_percentage / max(time_until_end[i], 1)
                
                charging_priorities.append((i, urgency, actions[i]))
        
        # sort cars by urgency
        charging_priorities.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        active_chargers = 0
        charged_cars = set()
        
        for priority_item in charging_priorities:
            i = priority_item[0]
            car = self.cars[i]
            
            car["charge_percentage"] = (car["charge"] / car["max_charge_kw"]) * 100
            
            car_wants_to_charge = (actions[i] == 1 and 
                                  car["charge_percentage"] < 100 and 
                                  time_until_end[i] > 0)
            
            can_charge = active_chargers < self.num_chargers
            
            if car_wants_to_charge and can_charge:
                charged_cars.add(i)
                
                if car["started_at"] is None:
                    car["started_at"] = self.t
                
                car["charge"] += car["charge_speed"]
                car["charge_percentage"] = min((car["charge"] / car["max_charge_kw"]) * 100, 100)
                car["charge_kw"].append(car["charge_speed"])
                
                # Count this car as using a charger
                active_chargers += 1
                
                # Give positive reward for charging progress - higher as deadline approaches
                progress_reward = car["charge_percentage"] / 50
                urgency_multiplier = 1 + (1 / max(time_until_end[i], 1))
                reward += progress_reward * urgency_multiplier
            else:
                # Not charging this hour, fill in the gap if charging has started
                if car["started_at"] is not None and time_until_end[i] > 0:
                    car["charge_kw"].append(0)
                
                # punish not charging when needed
                if time_until_end[i] <= 5 and car["charge_percentage"] < 90:
                    urgency_penalty = (100 - car["charge_percentage"]) / 10
                    reward -= urgency_penalty


        # calculate cost
        cost = sum([self.prices[self.t] * car["charge_speed"] if i < len(actions) and actions[i] > 0 and 
                    car["charge_percentage"] < 100 and time_until_end[i] > 0 and i < active_chargers else 0 
                    for i, car in enumerate(self.cars)])
        
        price_sensitivity = 1.0
        reward -= cost * price_sensitivity

        # Check end condition
        if self.t + 1 >= self.total_time:
            self.done = True
            for car in self.cars:
                if car["charge_percentage"] < car['min_percentage']:
                    reward -= 20 * (100 - car["charge_percentage"])
                if car["charge_percentage"] < 100:
                    reward -= 5 * (100 - car["charge_percentage"])
        
        # Check if any cars need to finalize their charging sessions
        for car in self.cars:
            if car["started_at"] is not None and (car["charge_percentage"] >= 100 or self.done):
                if car["started_at"] is not None:
                    self.schedule.append({
                        "id": car["id"],
                        "start_time": car["started_at"],
                        "charge": min(car["charge"], car['max_charge_kw']),
                        "charge_percentage": car["charge_percentage"],
                        "charge_kw": car["charge_kw"].copy()
                    })
                car["started_at"] = None
                car["charge_kw"] = []

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
    def __init__(self, state_dim, action_dim, max_cars, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.max_cars = max_cars

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.action_dim = action_dim

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                return int(torch.argmax(q_values).item())

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
            return True
        else:
            print(f"[red]No model found at {path}[/red]")
            return False


# ------------------------------
# Training Loop
# ------------------------------
def train_agent(env, agent, num_episodes=500, print_iter=False):
    update_target_every = 10  # update target network every 10 episodes
    for episode in range(num_episodes):
        state, _ = env.reset(env.num_cars, env.num_chargers)
        total_reward = 0
        done = False

        while not done:
            # Process each car individually
            action = np.zeros(env.max_cars, dtype=np.int32)
            for i in range(env.num_cars):
                action[i] = agent.select_action(state)
                
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            agent.replay_buffer.append((state, action[0], reward, next_state, float(done)))
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
    with open("./multi_train_progress", "w") as f:
        json.dump({"last_trained_index": last_index}, f)

def load_progress():
    if os.path.exists("./multi_train_progress"):
        with open("./multi_train_progress", "r") as f:
            return json.load(f).get("last_trained_index", -1)
    else:
        print("[yellow]The file does not exist, could not be loaded[/yellow]") 
        return -1

def remove_progress():
    if os.path.exists("./multi_train_progress"):
        os.remove("./multi_train_progress")
        os.remove("./multi_dqn_model_temp.pth")
    else:
        print("[yellow]The file does not exist, could not be removed[/yellow]") 

def train_dqn(cars: list[dict], rd: RequestDetail, num_chargers: int = None, num_cars: int = None, num_episodes=2000, period_length=48):
    if num_chargers is None:
        num_chargers = len(cars)
    if num_cars is None:
        num_cars = len(cars)
    data = EnergiData().call_api(rd)
    print(f"Days of data: {len(data)/24}")

    prices = [i.SpotPriceDKK / 1000 for i in data]
    times = [np.datetime64(i.HourDK) for i in data]

    prices_np = np.asarray(prices, dtype=np.float32)
    times_np = np.asarray(times, dtype=np.datetime64)

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

    if os.path.isfile("multi_dqn_model.pth"):
        finished = True

    if finished is False and not len(train_periods) == last_trained_index:
        print(f"Number of training periods: {len(train_periods)}")
        for i, (prices_48, times_48) in enumerate(train_periods):
            if i <= last_trained_index:
                continue  # Skip already trained periods

            print(f"\n[Training] Period {i+1}/{len(train_periods)} starting at {times_48[0]}")
            env = ElectricChargeEnv(prices_48, times_48, cars, num_chargers=num_chargers)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            if agent is None:
                agent = DQNAgent(state_dim, action_dim, env.max_cars, lr=1e-4)

            train_agent(env, agent, num_episodes=num_episodes)
            save_progress(i)

            agent.save("multi_dqn_model_temp.pth")

        if isinstance(agent, DQNAgent):
            # Save the trained model
            agent.save("multi_dqn_model.pth")

        remove_progress()

def run_dqn(cars: list[dict], rd: RequestDetail, num_chargers: int = None):
    if num_chargers is None:
        num_chargers = len(cars)
    data = EnergiData().call_api(rd)
    print(f"Days of data: {len(data)/24}")

    prices = [i.SpotPriceDKK / 1000 for i in data]
    times = [np.datetime64(i.HourDK) for i in data]

    prices_np = np.asarray(prices, dtype=np.float32)
    times_np = np.asarray(times, dtype=np.datetime64)

    prices_48 = prices_np[0:]
    times_48 = times_np[0:]

    env = ElectricChargeEnv(prices_48, times_48, cars, num_chargers=num_chargers)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, env.max_cars)
    
    model_loaded = agent.load("multi_dqn_model.pth")
    
    if model_loaded:
        print("\n[bold underline]Testing Trained Agent[/bold underline]")
        print(f"\n[Testing] starting at {times_48[0]}")
        state, _ = env.reset(num_cars=len(cars), num_chargers=num_chargers)
        done = False

        while not done:
            # Create actions array with fixed size
            action = np.zeros(env.max_cars, dtype=np.int32)
            
            # Process each car individually
            for i in range(min(len(cars), env.max_cars)):
                # Get state
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    q_values = agent.q_network(state_tensor)
                    action[i] = int(torch.argmax(q_values[0]).item())
            
            print(f"Step {env.t}/{len(times_48)}", end="\r")
            
            state, _, done, _, _ = env.step(action)

        if env.schedule:
            print(f"Schedule created with {len(env.schedule)} entries")
            print(env.schedule)
            return env.schedule
        else:
            print("[red]No schedule was generated[/red]")
            return "No schedule generated"
    else:
        return "No model trained"
