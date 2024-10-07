import numpy as np
import gymnasium as gym
from gymnasium import spaces
import xgboost as xgb
from stable_baselines3 import PPO
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
import shap
import pdb
import torch as th
import matplotlib.pyplot as plt


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
        return self.state, {}  # Return the state and an empty info dictionary

    def step(self, action):
        demand = self.state[0]
        time_slot = self.current_time
        previous_generation = self.state[2:]

        # Calculate new generation based on action
        generation = action * previous_generation  # Simplified model

        # Optimized reward function
        reward = optimized_reward_function(demand, generation, previous_generation,
                                           cost_per_unit=[10, 15, 12, 8, 20],
                                           ramp_up_cost=[2, 2, 2, 2, 2],
                                           ramp_down_cost=[2, 2, 2, 2, 2],
                                           start_up_cost=[500, 600, 550, 400, 700],
                                           shut_down_cost=[200, 250, 220, 180, 300])

        # Update the state
        self.current_time += 1
        self.state = np.concatenate(([demand, self.current_time], generation))
        done = self.current_time >= self.n_time_slots
        truncated = False  # Set this flag if the episode ended due to a time limit

        return self.state, reward, done, truncated, {}

    def render(self):
        pass

    def close(self):
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




def main():
    # Initialize environment
    env = DummyVecEnv([lambda: UnitCommitmentEnv(n_units=5, n_time_slots=10)])

    # Initialize PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=10000)

    # Test the trained model
    obs = env.reset()  # Adjusted to handle single return value
    for _ in range(10):
        action, _states = model.predict(obs)
        obs, reward, done, truncated = env.step(action)  # Updated to unpack 5 values
        if done:
            obs = env.reset()

        # Collect observations and action logits from the PPO model
    states = []
    logits = []

    obs = env.reset()[0]
    for _ in range(100):  # Collect 100 episodes of data
        action, _ = model.predict(obs)
        with th.no_grad():
            # Extract features from the mlp_extractor
            latent_pi = model.policy.mlp_extractor.policy_net(th.tensor(obs, dtype=th.float32))
            action_logits = model.policy.action_net(latent_pi)
        flat_obs = np.ravel(obs)
        states.append(flat_obs)

        logits.append(np.ravel(action_logits.numpy()))
        
        # Step in the environment
        obs, reward, done, truncated = env.step(action)
        if done:
            obs = env.reset()[0]

    # Convert the collected data into numpy arrays
    # size = [states[i].shape for i in range(100)]
    # pdb.set_trace()

    # Define a wrapper function to convert NumPy arrays to tensors
    def model_wrapper(input_data):
        input_tensor = th.tensor(input_data, dtype=th.float32)  # Convert to tensor
        return model.policy.mlp_extractor.policy_net(input_tensor).detach().numpy()

    states = np.array(states)
    logits = np.array(logits)
    states_tensor = th.tensor(states, dtype=th.float32)
    masker = shap.maskers.Independent(states)
    # Use SHAP to explain the actor's decision-making (action logits)
    explainer = shap.Explainer(model_wrapper, masker)
    shap_values = explainer(states)
    # pdb.set_trace()
    shap_values_mean = shap_values.mean(axis=2)
    # Plot the SHAP values for a particular state
    shap.summary_plot(shap_values_mean, states)
    plt.savefig("shap_summary_plot.png")

    
    # SHAP Explainability
    obs_data = []
    obs = env.reset()[0]
    pdb.set_trace()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, reward, done, truncated = env.step(action)  # Updated to unpack 5 values
        # pdb.set_trace()
        obs_data.append(obs[0])
        if done:
            obs = env.reset()[0]

    headers = ["demand", "time_slots", "gen_1", "gen_2", "gen_3", "gen_4", "gen_5"]

    obs_data = np.array(obs_data)
    obs_data_torch = th.tensor(obs_data, dtype=th.float32)
    obs_data_np = obs_data_torch.numpy()
    obs_data_pd = pd.DataFrame(obs_data_np, columns=headers)
    pdb.set_trace()
    model.policy.actor.save_model('policy.json')

# Load the model from the saved binary file
# loaded_model = xgb.XGBRegressor()
# loaded_model.load_model('model.json')
    # pdb.set_trace()
    # masker = shap.maskers.Independent(data=obs_data, max_samples=100)
    # Explaining model's decisions using SHAP

    masker = shap.maskers.Independent(obs_data_np)
    explainer = shap.Explainer(model.policy, masker)
    shap_values = explainer(obs_data_pd)

    # Plot the SHAP values
    shap.summary_plot(shap_values, obs_data)

    # Visualize the learned policy
    obs = env.reset()
    for t in range(10):
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)  # Updated to unpack 5 values
        plt.plot(action, label=f'Time Slot {t}')

    plt.xlabel('Unit Index')
    plt.ylabel('Generation Level')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
