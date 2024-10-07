from env import UnitCommitmentEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import shap
import numpy as np
import matplotlib.pyplot as plt

def main():
    env = DummyVecEnv([lambda: UnitCommitmentEnv(n_units=5, n_time_slots=10)])

    # Initialize PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=10000)

    # Test the trained model
    obs, _ = env.reset()
    for _ in range(10):
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()

    # SHAP Explainability
    obs_data = []
    obs, _ = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        obs_data.append(obs)
        if done:
            obs, _ = env.reset()

    obs_data = np.array(obs_data)

    # Explaining model's decisions using SHAP
    explainer = shap.Explainer(model.policy)
    shap_values = explainer(obs_data)

    # Plot the SHAP values
    shap.summary_plot(shap_values, obs_data)

    # Visualize the learned policy
    obs, _ = env.reset()
    for t in range(10):
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        plt.plot(action, label=f'Time Slot {t}')

    plt.xlabel('Unit Index')
    plt.ylabel('Generation Level')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
