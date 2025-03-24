import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Define the RL Environment
class EVChargingEnv:
    def __init__(self, prices, num_cars=5, num_chargers=3, charge_rate=3, battery_capacity=10):
        self.prices = prices  # Electricity price for each hour
        self.num_cars = num_cars
        self.num_chargers = num_chargers
        self.charge_rate = charge_rate  # Units of charge per hour
        self.battery_capacity = battery_capacity  # Max battery capacity
        self.time = 0  # Start at hour 0
        self.car_battery = np.zeros(num_cars)  # Battery levels of each car
        self.max_time = len(prices)  # Total hours available for charging

    def reset(self):
        self.time = 0
        self.car_battery = np.zeros(self.num_cars)
        return self._get_state()

    def _get_state(self):
        return np.concatenate(([self.time-1], self.car_battery, [self.prices[self.time-1]]))

    def step(self, action):
        chargeable_cars = np.where(action == 1)[0]
        num_active_chargers = min(len(chargeable_cars), self.num_chargers)
        cost = np.sum(self.prices[self.time] * num_active_chargers)
        
        for car in chargeable_cars[:num_active_chargers]:
            self.car_battery[car] = min(self.battery_capacity, self.car_battery[car] + self.charge_rate)  # Charge cars up to max capacity
        
        reward = -cost  # Minimize cost
        self.time += 1
        done = self.time >= self.max_time
        return self._get_state(), reward, done, {}

    def get_optimal_charging_time(self):
        optimal_time = np.argmin(self.prices)
        min_price = self.prices[optimal_time]
        charge_time = self.battery_capacity / self.charge_rate
        print(f"Optimal charging time is at hour {optimal_time} with price {min_price:.2f}. It will take approximately {charge_time:.1f} hours to fully charge a car.")

# Define the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Training Loop
def train_dqn(env, num_episodes=1000, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, lr=0.001):
    input_dim = len(env._get_state())
    output_dim = env.num_cars  # Each car can be charged or not (binary actions)
    model = DQN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    replay_memory = deque(maxlen=10000)
    
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0
        done = False
        while not done:
            if random.random() < epsilon:
                action = np.random.randint(0, 2, size=output_dim)  # Random action (exploration)
            else:
                q_values = model(state)
                action = (q_values > 0).cpu().numpy().astype(int)  # Select actions

            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)
            replay_memory.append((state, torch.FloatTensor(action), reward, next_state, done))
            state = next_state
            total_reward += reward
            
            # Train the model
            if len(replay_memory) > 32:
                batch = random.sample(replay_memory, 32)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
                state_batch = torch.stack(state_batch)
                action_batch = torch.stack(action_batch)
                reward_batch = torch.FloatTensor(reward_batch)
                next_state_batch = torch.stack(next_state_batch)
                done_batch = torch.BoolTensor(done_batch)
                
                q_values = model(state_batch)
                next_q_values = model(next_state_batch).detach()
                target_q_values = reward_batch + gamma * next_q_values.max(dim=1)[0] * ~done_batch
                loss = criterion(q_values.max(dim=1)[0], target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = max(0.1, epsilon * epsilon_decay)  # Decay exploration rate
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

# Example Usage
if __name__ == "__main__":
    num_hours = 24  # Next two days
    electricity_prices = np.random.uniform(0.1, 0.5, num_hours)  # Simulated prices
    env = EVChargingEnv(electricity_prices, num_cars=5, num_chargers=3)
    train_dqn(env)
    env.get_optimal_charging_time()
