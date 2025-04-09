from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
import torch.optim as optim
import random
from collections import deque
from rich import print
from dateutil import parser

from apis.EnergiData import EnergiData, RequestDetail
 


# ------------------------------
# Custom Environment Definition
# ------------------------------
class ElectricChargeEnv:
    def __init__(self, prices, num_cars, num_chargers):
        """
        prices: list or np.array of electricity prices for 48 hours (the forecast)
        num_cars: total number of cars that need charging
        num_chargers: available number of chargers per time step
        """
        self.prices = np.array(prices)
        self.num_cars = num_cars
        self.num_chargers = num_chargers
        self.total_time = len(prices)
        self.max_price = np.max(self.prices)
        self.reset()

    def reset(self):
        self.t = 0
        # Create a list of car dictionaries with an ID and charging status.
        self.cars = [{'id': i, 'charged': False, 'charge_time': None} for i in range(self.num_cars)]
        self.uncharged_car_ids = list(range(self.num_cars))
        self.done = False
        self.schedule = []  # list of tuples: (time, list of car ids charged in that time step)
        return self._get_state()

    def _get_state(self):
        # State: [normalized current time, remaining_cars ratio, full normalized price forecast]
        norm_time = self.t / (self.total_time - 1)
        norm_remaining = len(self.uncharged_car_ids) / self.num_cars
        norm_prices = self.prices / (self.max_price + 1e-6)
        state = np.concatenate(([norm_time, norm_remaining], norm_prices))
        return state.astype(np.float32)

    def step(self, action):
        """
        action: integer from 0 to num_chargers.
        If action > number of remaining cars, it is clipped.
        """
        valid_action = min(action, len(self.uncharged_car_ids), self.num_chargers)
        # Compute cost for charging valid_action cars this hour.
        cost = self.prices[self.t] * valid_action
        reward = -cost  # negative cost as reward

        # Mark cars as charged.
        cars_charged_now = []
        for _ in range(valid_action):
            car_id = self.uncharged_car_ids.pop(0)
            self.cars[car_id]['charged'] = True
            self.cars[car_id]['charge_time'] = self.t
            cars_charged_now.append(car_id)
        # Record schedule: which car ids got charged at hour self.t.
        self.schedule.append((self.t, cars_charged_now))
        
        self.t += 1

        # Check for termination: either time is up or all cars are charged.
        if self.t >= self.total_time:
            self.done = True
            # Penalty for any uncharged cars.
            if self.uncharged_car_ids:
                reward -= 10 * len(self.uncharged_car_ids)
        elif not self.uncharged_car_ids:
            self.done = True

        return self._get_state(), reward, self.done, {}

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
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64

    def select_action(self, state):
        # Epsilon-greedy action selection.
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    @property
    def action_dim(self):
        # Action dimension is defined by the network output dimension.
        return self.q_network.net[-1].out_features

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
        loss = nn.MSELoss()(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# ------------------------------
# Training Loop
# ------------------------------
def train_agent(env, agent, num_episodes=500):
    update_target_every = 10  # update target network every 10 episodes
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            agent.replay_buffer.append((state, action, reward, next_state, float(done)))
            state = next_state

            agent.update()
        
        # Decay epsilon after each episode.
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)

        if episode % update_target_every == 0:
            agent.update_target()

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    return agent
 
# ------------------------------
# Example Usage
# ------------------------------
def run():
    # For demonstration, create a dummy price forecast for 48 hours.
    # np.random.seed(0)
    # prices = np.random.uniform(low=0.2, high=0.8, size=48)
    rd = RequestDetail(
        startDate="StartOfDay-P5D",
        dataset="Elspotprices",
        filter_json=json.dumps({"PriceArea": ["DK1"]}),
        # optional= "HourDK,SpotPriceDKK",
        limit=0
    )
    data = EnergiData().call_api(rd)
    prices = []
    times = []
    for i in data:
        prices.append(i.SpotPriceDKK/1000)
        times.append(parser.parse(i.HourDK))
    # print(prices)
    # print(times)
    prices_np = np.asarray(prices, dtype=np.float32)
    times_np = np.asarray(times, dtype=np.datetime64)
    # print(prices_np)
    # print(times_np)

    # Set the number of cars and chargers.
    num_cars = 3      # e.g., 10 cars to charge
    num_chargers = 1   # e.g., 3 chargers available each hour

    # Create the environment.
    env = ElectricChargeEnv(prices_np, num_cars, num_chargers)

    # Define state dimension: [normalized time, normalized remaining cars] + 48 normalized prices.
    # state_dim = 2 + len(prices)
    data_points = prices_np
    print(len(data_points))
    while len(data_points) > 0:
        state_dim = 2 + len(data)
        # Define action dimension as (num_chargers + 1) because we can choose to charge 0...num_chargers cars.
        action_dim = num_chargers + 1

        # Create the DQN agent.
        agent = DQNAgent(state_dim, action_dim, lr=1e-5)

        # Train the agent.
        print("Training agent...")
        env.prices = data_points[:49]
        data_points = prices_np[48:]
        trained_agent = train_agent(env, agent, num_episodes=400)


    # Test the trained agent on a new episode using a greedy policy.
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(trained_agent.device)
        with torch.no_grad():
            q_values = trained_agent.q_network(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        state, reward, done, _ = env.step(action)

    # Display the charging schedule per car.
    print("\nOptimal Charging Schedule (per hour):")
    for hour, car_ids in env.schedule:
        if car_ids:
            car_list = ", ".join([f"Car {cid}" for cid in car_ids])
            print(f"At hour {hour} (Price: {prices_np[hour]:.2f}) -> Charged: {car_list}")

    # Optionally, print a summary for each car.
    print("\nCharging Summary per Car:")
    for car in env.cars:
        if car['charged']:
            print(f"Car {car['id']} charged at hour {car['charge_time']}")
        else:
            print(f"Car {car['id']} was not charged!")
