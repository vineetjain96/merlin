from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces


class Nav2dEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None):
        super(Nav2dEnv, self).__init__()

        self.observation_space = spaces.Box(low=np.array([-10.0, -10.0]), high=np.array([10.0, 10.0]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self.max_episode_steps = 100
        self.num_steps = None
        self.goal = None
        self.state = None
        self.eps = 0.5  # Tolerance for the goal

        self.path = []


    def step(self, action):
        self.num_steps += 1

        # normalize action to be unit vector
        action = action / np.linalg.norm(action, axis=-1, keepdims=True)
        delta_x = action[0]
        delta_y = action[1]
        new_state = self.state + np.array([delta_x, delta_y])

        # Clip the new state within the state space boundaries
        self.state = np.clip(new_state, self.observation_space.low, self.observation_space.high)
        
        # Define a simple reward function
        distance = np.linalg.norm(self.state - self.goal)
        reward = 1.0 if distance < self.eps else 0.0

        # Check if the episode is done (you can define your termination conditions)
        terminated = distance < self.eps
        truncated = self.num_steps >= self.max_episode_steps

        self.path.append(self.state)

        # Additional information (optional)
        info = {}

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, info

    def reset(self):
        self.num_steps = 0
        # Reset the agent's state to a random position within the state space
        
        #self.goal = np.random.uniform(self.observation_space.low, self.observation_space.high)
        self.goal = np.array([0., 0.])
        self.state = np.random.uniform(self.observation_space.low, self.observation_space.high)
        self.path = [self.state]

        info = {'goal': np.array(self.goal, dtype=np.float32)}

        return np.array(self.state, dtype=np.float32), info

    def render(self, mode='human'):
        if mode == 'human':
            plt.figure(figsize=(6, 6))
            plt.plot([p[0] for p in self.path], [p[1] for p in self.path], marker='o', linestyle='-', color='b')
            plt.plot(self.goal[0], self.goal[1], marker='x', color='r', markersize=10)
            plt.title("Agent's Path")
            plt.xlim(self.observation_space.low[0], self.observation_space.high[0])
            plt.ylim(self.observation_space.low[1], self.observation_space.high[1])
            plt.xlabel('X')
            plt.ylabel('Y')
            #plt.grid()
            plt.savefig('nav2d.png')
            plt.close()


    def close(self):
        # Clean-up code (optional)
        pass


gym.register("Nav2d-v0", entry_point=Nav2dEnv)

# Test the environment
if __name__ == "__main__":
    env = gym.make("Nav2d-v0")
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()  # Replace with your policy
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        env.render()

    print("Total reward:", total_reward)
    env.close()
