from time import strftime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import json
from rich import print
from dateutil import parser

from apis.EnergiData import EnergiData, RequestDetail
 
import gymnasium as gym
from gymnasium import spaces

class ElectricChargeEnv(gym.Env):
    """
    Custom environment for optimizing electric car charging schedules.
    """
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, prices, num_cars, num_chargers):
        super(ElectricChargeEnv, self).__init__()
        
        self.prices = np.array(prices, dtype=np.float32)
        self.num_cars = num_cars
        self.num_chargers = num_chargers
        self.total_time = len(prices)
        self.max_price = np.max(self.prices)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(num_chargers + 1)  # Charge 0 to num_chargers cars
        self.observation_space = spaces.Box(low=0, high=1, shape=(2 + self.total_time,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.t = 0
        self.cars = [{'id': i, 'charged': False, 'charge_time': None} for i in range(self.num_cars)]
        self.uncharged_car_ids = list(range(self.num_cars))
        self.done = False
        self.schedule = []  # (time, list of car ids charged at that time)
        
        return self._get_state(), {}

    def _get_state(self):
        """Returns the current state as a normalized feature vector."""
        norm_time = self.t / (self.total_time - 1)
        norm_remaining = len(self.uncharged_car_ids) / self.num_cars
        norm_prices = self.prices / (self.max_price + 1e-6)
        return np.concatenate(([norm_time, norm_remaining], norm_prices)).astype(np.float32)

    def step(self, action):
        """Executes a step in the environment given an action."""
        valid_action = min(action, len(self.uncharged_car_ids), self.num_chargers)
        cost = self.prices[self.t] * valid_action
        reward = -cost  # Minimize cost (negative reward)

        cars_charged_now = []
        for _ in range(valid_action):
            car_id = self.uncharged_car_ids.pop(0)
            self.cars[car_id]['charged'] = True
            self.cars[car_id]['charge_time'] = self.t
            cars_charged_now.append(car_id)

        self.schedule.append((self.t, cars_charged_now))
        self.t += 1

        if self.t >= self.total_time:
            self.done = True
            if self.uncharged_car_ids:
                reward -= 10 * len(self.uncharged_car_ids)  # Penalty for uncharged cars
        elif not self.uncharged_car_ids:
            self.done = True

        return self._get_state(), reward, self.done, False, {}

    def render(self, mode="human"):
        """Optional render method to visualize the schedule."""
        print(f"Time Step {self.t}: Schedule -> {self.schedule}")

    def close(self):
        """Clean up if necessary."""
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
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
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
        # Epsilon-greedy action selection
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


# ------------------------------
# Training Loop
# ------------------------------
def train_agent(env, agent, num_episodes=500):
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
# Example Usage
# ------------------------------
def run():
    """Runs the training and testing process for the electric charging environment."""
    rd = RequestDetail(
        startDate="StartOfYear",
        dataset="Elspotprices",
        filter_json=json.dumps({"PriceArea": ["DK1"]}),
        limit=0
    )
    data = EnergiData().call_api(rd)
    print(f"Days of data: {len(data)}")

    prices = [i.SpotPriceDKK / 1000 for i in data]
    times = [parser.parse(i.HourDK) for i in data]

    prices_np = np.asarray(prices, dtype=np.float32)
    times_np = np.asarray(times, dtype=np.datetime64)

    # Train on rolling 48-hour periods, ensuring each 24-hour period is used at most twice
    for start_idx in range(0, len(prices_np) - 47, 24):

        prices_48 = prices_np[start_idx:start_idx + 48]
        times_48 = times_np[start_idx:start_idx + 48]

        # Set the number of cars and chargers
        num_cars = 3
        num_chargers = 1

        # Create the environment
        env = ElectricChargeEnv(prices_48, num_cars, num_chargers)

        # Define state and action dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Create the DQN agent
        agent = DQNAgent(state_dim, action_dim, lr=1e-4)

        # Train the agent
        print(f"\nTraining agent for period starting at {times_48[0]}")
        trained_agent = train_agent(env, agent, num_episodes=1000)

        # Test the trained agent on a new episode using a greedy policy
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(trained_agent.device)
            with torch.no_grad():
                q_values = trained_agent.q_network(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
            state, reward, done, _, _ = env.step(action)

    print("\nOptimal Charging Schedule (per hour):")
    for hour, car_ids in env.schedule:
        if car_ids:
            car_list = ", ".join([f"Car {cid}" for cid in car_ids])
            charge_time = times_48[hour]
            # Convert numpy.datetime64 to a Python datetime object for strftime
            charge_time_dt = charge_time.astype('M8[s]').tolist()  # Convert to seconds resolution
            charge_time_str = charge_time_dt.strftime("%Y-%m-%d %H:%M")
            print(f"At {charge_time_str} (Price: {prices_48[hour]:.2f}) -> Charged: {car_list} (Hour {hour})")

