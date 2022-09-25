import gym
import gym_snake
import numpy as np


from  stable_baselines3 import PPO
import stable_baselines3 as sb3

env = gym.make('Snake-8x8-v0')

model = PPO("MlpPolicy", env, n_steps=512, verbose=1, seed=20)

# Train the agent for `num_steps` steps

model.learn(total_timesteps=1000, eval_env=env, eval_freq=100)  # change 1 to 10000 (prod)

print("Learning complete")

# Evaluate the trained agent
mean_reward, std_reward = sb3.common.evaluation.evaluate_policy(model, env,
                                                                    n_eval_episodes=20)  # change 1 to 100 (prod)
