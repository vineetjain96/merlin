import os
import numpy as np
import argparse
import torch
import gymnasium as gym

import torch
import torch.nn as nn

from nav2d_env import Nav2dEnv
from plotter import plot_mesh
from src.components.buffer import ReplayBuffer


class MLP(nn.Module):
    def __init__(self, input_size, \
                 hidden_size, output_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2*input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x, g, noise=0):
        x = torch.cat([x, g], dim=-1)
        x = self.network(x)
        x += torch.randn_like(x) * noise
        return x


class GCSL:
    def __init__(self,
        env,
        device,
        max_timesteps=1e5,
        explore_timesteps=1e3,
        max_path_length=50,
        start_policy_timesteps=0,
        batch_size=100,
        n_accumulations=1,
        policy_updates_per_step=10,
        train_policy_freq=None,
        lr=5e-4,
    ):
        self.env = env
        self.device = device

        self.state_space = env.observation_space
        self.action_space = env.action_space
        self.state_shape = self.state_space.shape
        self.state_dtype = self.state_space.dtype
        self.action_shape = self.action_space.shape[0]
        self.action_dtype = self.action_space.dtype
        self.goal_shape = self.state_shape[0]
        self.goal_dtype = self.state_dtype


        self.policy = MLP(self.state_shape[0], 256, self.action_shape).to(self.device)
        self.replay_buffer = ReplayBuffer(
            self.state_shape, self.state_dtype, \
            self.action_shape, self.action_dtype, \
            self.goal_shape, self.goal_dtype, \
            max_path_length, her_ratio=1.0,
            buffer_size=20000,
        )

        self.max_timesteps = max_timesteps
        self.explore_timesteps = explore_timesteps
        self.max_path_length = max_path_length

        self.start_policy_timesteps = start_policy_timesteps

        if train_policy_freq is None:
            train_policy_freq = self.max_path_length

        self.train_policy_freq = train_policy_freq
        self.batch_size = batch_size
        self.n_accumulations = n_accumulations
        self.policy_updates_per_step = policy_updates_per_step
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)


    def loss_fn(self, observations, goals, actions, horizons, weights):
        obs_dtype = torch.float32
        action_dtype = torch.float32

        observations_torch = torch.tensor(observations, dtype=obs_dtype).to(self.device)
        goals_torch = torch.tensor(goals, dtype=obs_dtype).to(self.device)
        actions_torch = torch.tensor(actions, dtype=action_dtype).to(self.device)
        actions_pred = self.policy(observations_torch, goals_torch)
        loss = torch.nn.MSELoss()

        return loss(actions_pred, actions_torch)
    
    def sample_trajectory(self):
        states = []
        actions = []

        with torch.no_grad():
            state, info = self.env.reset()
            states.append(state)
            goal = info['goal']
            goal_torch = torch.from_numpy(np.array(goal, dtype=np.float32)).to(self.device)
            for t in range(1, self.max_path_length):
                state_torch = torch.from_numpy(np.array(state, dtype=np.float32)).to(self.device)
                action = self.policy(state_torch, goal_torch, \
                                      noise=0.1).cpu().numpy()
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                state, _, _, _, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
        
        return np.stack(states), np.array(actions), goal
    
    def sample_random_trajectory(self):            
        states = []
        actions = []

        with torch.no_grad():
            state, info = self.env.reset()
            states.append(state)
            goal = info['goal']
            for t in range(1, self.max_path_length):
                action = np.random.uniform(self.env.action_space.low, \
                                        self.env.action_space.high)
                state, _, _, _, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
        
        return np.stack(states), np.array(actions), goal

    def take_policy_step(self, buffer=None):
        if buffer is None:
            buffer = self.replay_buffer

        avg_loss = 0
        self.policy_optimizer.zero_grad()
    
        for _ in range(self.n_accumulations):
            observations, actions, goals, _, horizons, weights = buffer.sample_batch(self.batch_size)
            loss = self.loss_fn(observations, goals, actions, horizons, weights)
            loss.backward()
            avg_loss += loss.item()
        
        self.policy_optimizer.step()
        
        return avg_loss / self.n_accumulations
        
    def train(self):
        # Evaluate untrained policy
        total_timesteps = 0
        timesteps_since_train = 0
        timesteps_since_eval = 0

        running_loss = None
        
        while total_timesteps < self.max_timesteps:
            if total_timesteps < self.explore_timesteps:
                states, actions, goal_state = self.sample_random_trajectory()
            else:
                states, actions, goal_state = self.sample_trajectory()
            achieved_states = states

            self.replay_buffer.add_trajectory(states, actions, goal_state, achieved_states)

            total_timesteps += self.max_path_length
            timesteps_since_train += self.max_path_length
            timesteps_since_eval += self.max_path_length

            # Take training steps
            if timesteps_since_train >= self.train_policy_freq and total_timesteps > self.start_policy_timesteps:
                timesteps_since_train %= self.train_policy_freq
                self.policy.train()
                for _ in range(int(self.policy_updates_per_step * self.train_policy_freq)):
                    loss = self.take_policy_step()
                    if running_loss is None:
                        running_loss = loss
                    else:
                        running_loss = 0.9 * running_loss + 0.1 * loss

            if total_timesteps % 100 == 0:
                print("Timesteps: {}, Loss: {}".format(total_timesteps, running_loss))
            
        plot_mesh(self.env, self.policy, self.device, \
                    name='gcsl_{}'.format(total_timesteps), \
                    gcsl=True)
        plot_mesh(self.env, self.policy, self.device, \
                name='gcsl_ood_{}'.format(total_timesteps), \
                gcsl=True, ood=True)

def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    # env args
    parser.add_argument("--max-timesteps", type=int, default=1e5)
    parser.add_argument("--max-path-length", type=int, default=50)
    return parser.parse_args()

if __name__ == "__main__":
    env = gym.make("Nav2d-v0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = get_args()
    model = GCSL(env, device, max_timesteps=args.max_timesteps, max_path_length=args.max_path_length)
    model.train()    