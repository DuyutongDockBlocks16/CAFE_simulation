import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

def policy(observation):
    return env.action_space.sample()


env = gym.make("FetchPickAndPlace-v3", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()