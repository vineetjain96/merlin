import os
import numpy as np
import math
import argparse
import gymnasium as gym

import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

from nav2d_env import Nav2dEnv
from plotter import plot_mesh, plot_noisy_trajectories
from src.components.buffer import ReplayBuffer


def timestep_embedding(timesteps, dim=32, max_period=50):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    if len(timesteps.shape) == 1:
        args = timesteps.float() * freqs
    else:
        args = timesteps.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MLP(nn.Module):
    def __init__(self, input_size, timestep_embedding_size,
                hidden_size, output_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2*input_size+timestep_embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x, g, t):
        t = timestep_embedding(t, 32, 50)
        x = torch.cat([x, g, t], dim=-1)
        x = self.network(x)
        mu, kappa = torch.split(x, 2, dim=-1)
        kappa = torch.nn.functional.softplus(kappa)
        return mu, kappa

class Merlin:
    def __init__(self, 
        env, 
        device, 
        max_timesteps=1e5,
        max_path_length=50,
        test_horizon=1,
    ):
        self.env = env
        self.device = device
        self.max_timesteps = max_timesteps
        self.max_path_length = max_path_length
        self.test_horizon = test_horizon

        self.state_space = env.observation_space
        self.action_space = env.action_space
        self.state_shape = self.state_space.shape
        self.state_dtype = self.state_space.dtype
        self.action_shape = self.action_space.shape[0]
        self.action_dtype = self.action_space.dtype
        self.goal_shape = self.state_shape[0]
        self.goal_dtype = self.state_dtype

        self.policy = MLP(self.state_shape[0], 32, 256, 2*self.action_shape).to(self.device)
        self.replay_buffer = ReplayBuffer(
            self.state_shape, self.state_dtype, \
            self.action_shape, self.action_dtype, \
            self.goal_shape, self.goal_dtype, \
            max_path_length, her_ratio=1.0,
            buffer_size=20000,
        )

        self.policy_updates_per_step = None
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-5)
        self.batch_size = 256

    def forward_noise(self, state, t):
        states = []
        actions = []
        achieved_states = []
        for _ in range(1, t):
            action = np.random.uniform(self.env.action_space.low, \
                                        self.env.action_space.high)
            action = action / np.linalg.norm(action, axis=-1, keepdims=True)
            delta_x = action[0]
            delta_y = action[1]
            new_state = state - np.array([delta_x, delta_y])
            state = np.clip(new_state, self.env.observation_space.low, self.env.observation_space.high)
            states.append(state)
            actions.append(action)
            achieved_states.append(state)
        
        states.append(state)
        achieved_states.append(state)

        states.reverse()
        actions.reverse()
        achieved_states.reverse()
        
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), \
            np.array(achieved_states, dtype=np.float32)

    def train(self):
        if self.policy_updates_per_step is None:
            self.policy_updates_per_step = self.max_path_length * 10

        total_timesteps = 0
        train_losses = []
        self.policy.train()

        pairs = np.column_stack(np.triu_indices(self.max_path_length, k=1))
        np.random.shuffle(pairs)
        
        noisy_states, noisy_actions = [], []
        while total_timesteps < self.max_timesteps:
            _, info = self.env.reset()
            goal = info['goal']

            states, actions, achieved_states = self.forward_noise(goal, self.max_path_length)
            noisy_states.append(states)

            total_timesteps += self.max_path_length
            
            self.replay_buffer.add_trajectory(states, actions, goal, achieved_states)
            _losses = []

            for _ in range(self.policy_updates_per_step):
                traj_idxs = np.random.choice(self.replay_buffer.current_buffer_size, self.batch_size)
                time_idxs = np.random.choice(pairs.shape[0], self.batch_size)
                pairs_batch = pairs[time_idxs]

                _states, _actions, _goals, _, _horizons, _ = \
                    self.replay_buffer._get_batch(traj_idxs, pairs_batch[:, 0], pairs_batch[:, 1])
                _states = torch.from_numpy(np.array(_states, dtype=np.float32)).to(self.device)
                _goals = torch.from_numpy(np.array(_goals, dtype=np.float32)).to(self.device)
                _actions = torch.from_numpy(np.array(_actions, dtype=np.float32)).to(self.device)
                _horizons = torch.tensor(pairs_batch[:, 1] - pairs_batch[:, 0])[:, None].float().to(self.device)
    
                mus, kappas = self.policy(_states, _goals, _horizons)
                dist = Independent(Normal(mus, kappas), 1)
                loss = -dist.log_prob(_actions).mean()

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                _losses.append(np.mean(np.array(loss.item())))
            
            train_losses.append(np.mean(np.array(_losses)))
            
            if total_timesteps % 100 == 0:
                print("Timesteps: {}, Loss: {}".format(total_timesteps, np.mean(np.array(loss.item()))))
            
        for test_horizon in [1, 5, 10, 20, 50, 200]:
            plot_mesh(self.env, self.policy, self.device, \
                    name='merlin_{}_t_{}'.format(total_timesteps, test_horizon), \
                    test_horizon=test_horizon)
            plot_mesh(self.env, self.policy, self.device, \
                    name='merlin_ood_{}_t_{}'.format(total_timesteps, test_horizon), \
                    test_horizon=test_horizon, ood=True)
        
        plot_noisy_trajectories(self.env, noisy_states, self.max_path_length, name='diff_trajectories')


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    # env args
    parser.add_argument("--max-timesteps", type=int, default=1e4)
    parser.add_argument("--max-path-length", type=int, default=50)
    parser.add_argument("--test-horizon", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    env = gym.make("Nav2d-v0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = get_args()
    model = Merlin(env, device, args.max_timesteps, args.max_path_length, args.test_horizon)
    model.train()
