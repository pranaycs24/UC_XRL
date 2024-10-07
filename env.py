import numpy as np
import gymnasium as gym
from gymnasium import spaces

class UnitCommitmentEnv(gym.Env):
    def __init__(self, n_units, n_time_slots):
        super(UnitCommitmentEnv, self).__init__()
        self.n_units = n_units
        self.n_time_slots = n_time_slots
        self.action_space = spaces.Box(low=0, high=1, shape=(n_units,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(n_units + 2,), dtype=np.float32)
        self.state = None
        self.current_time = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0
        demand = np.random.uniform(low=50, high=100)  # Example demand range
        time_slot = 0
        previous_generation = np.zeros(self.n_units)
        self.state = np.concatenate(([demand, time_slot], previous_generation))
        return self.state, {}

    def step(self, action):
        demand = self.state[0]
        time_slot = self.state[1]
        previous_generation = self.state[2:]

        # Calculate new generation based on action
        generation = action * previous_generation  # Simplified model

        reward = optimized_reward_function(demand, generation, previous_generation,
                                           cost_per_unit=[10, 15, 12, 8, 20],
                                           ramp_up_cost=[2, 2, 2, 2, 2],
                                           ramp_down_cost=[2, 2, 2, 2, 2],
                                           start_up_cost=[500, 600, 550, 400, 700],
                                           shut_down_cost=[200, 250, 220, 180, 300])

        # Update the state
        self.current_time += 1
        self.state = np.concatenate(([demand, time_slot], generation))
        done = time_slot >= self.n_time_slots
        return self.state, reward, done, {}

    def render(self):
        pass


def calculate_cost(generation, cost_per_unit):
    return np.sum(generation * cost_per_unit)


def calculate_ramp_penalty(current_gen, previous_gen, ramp_up_cost, ramp_down_cost):
    ramp_up = np.maximum(current_gen - previous_gen, 0)
    ramp_down = np.maximum(previous_gen - current_gen, 0)
    return np.sum(ramp_up * ramp_up_cost) + np.sum(ramp_down * ramp_down_cost)

def optimized_reward_function(demand, generation, previous_generation, cost_per_unit, ramp_up_cost, ramp_down_cost, start_up_cost, shut_down_cost):
    total_generation = np.sum(generation)
    
    # Insufficient generation penalty
    if total_generation < demand:
        reward = -1000 * (demand - total_generation)  # Heavy penalty
    else:
        reward = 0
    
    # Excessive generation penalty (e.g., overgeneration)
    if total_generation > demand:
        reward -= 50 * (total_generation - demand)
    
    # Cost of generation
    reward -= calculate_cost(generation, cost_per_unit)
    
    # Ramp-up/Ramp-down penalties
    reward -= calculate_ramp_penalty(generation, previous_generation, ramp_up_cost, ramp_down_cost)
    
    # Start-up/Shut-down costs (assuming binary status for simplicity)
    for i in range(len(generation)):
        if previous_generation[i] == 0 and generation[i] > 0:  # Start-up
            reward -= start_up_cost[i]
        elif previous_generation[i] > 0 and generation[i] == 0:  # Shut-down
            reward -= shut_down_cost[i]
    
    return reward

