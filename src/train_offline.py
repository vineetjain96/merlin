import os
import pickle
import math
import argparse
import time

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Independent

from src.envs import register_envs
from src.components.buffer import ReplayBuffer
from src.components.dynamics_model import train_dynamics, train_reverse_bc
from src.components.representations import train_contrastive_encoder
from src.components.normalizer import normalizer
from src.components.trajectory_stitching import StateTree


HER_RATIO_TUNED = {
    'PointReach': {'expert': 1.0, 'random': 1.0},
    'PointRooms': {'expert': 0.2, 'random': 1.0},
    'Reacher': {'expert': 0.2, 'random': 1.0},
    'SawyerReach': {'expert': 1.0, 'random': 1.0},
    'SawyerDoor': {'expert': 0.2, 'random': 1.0},
    'FetchReach': {'expert': 1.0, 'random': 1.0},
    'FetchPush': {'expert': 0.0, 'random': 0.2},
    'FetchPick': {'expert': 0.0, 'random': 0.5},
    'FetchSlide': {'expert': 0.2, 'random': 0.8},
    'HandReach': {'expert': 1.0, 'random': 1.0},
}

HORIZON_TUNED = {
    'PointReach': {'expert': 1, 'random': 1},
    'PointRooms': {'expert': 1, 'random': 1},
    'Reacher': {'expert': 5, 'random': 5},
    'SawyerReach': {'expert': 1, 'random': 1},
    'SawyerDoor': {'expert': 5, 'random': 5},
    'FetchReach': {'expert': 1, 'random': 1},
    'FetchPush': {'expert': 20, 'random': 20},
    'FetchPick': {'expert': 10, 'random': 50},
    'FetchSlide': {'expert': 10, 'random': 10},
    'HandReach': {'expert': 1, 'random': 1},
}

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


class Policy(nn.Module):
    def __init__(self, state_shape, goal_dim,\
                 timestep_embedding_dim, hidden_dim, output_dim, \
                 max_path_length, max_action):
        super(Policy, self).__init__()
        self.timestep_embedding_dim = timestep_embedding_dim
        self.output_dim = output_dim
        self.max_path_length = max_path_length
        self.max_action = max_action
        if len(state_shape) == 1:
            self.cnn = False
            state_dim = state_shape[0]
            self.conv_layers = nn.Identity()
            self.dense_layers = nn.Sequential(
                nn.Linear(state_dim+goal_dim+timestep_embedding_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.cnn = True
            h, w, c = state_shape
            self.conv_layers = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Flatten(-3, -1),
            )
            with torch.no_grad():
                embedding_dim = int(np.prod(self.conv_layers(torch.zeros(1, c, h, w)).shape[1:]))
            self.dense_layers = nn.Sequential(
                nn.Linear(embedding_dim+goal_dim+timestep_embedding_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )


    def forward(self, x, g, t):
        x = self.conv_layers(x)
        t = timestep_embedding(t, self.timestep_embedding_dim, self.max_path_length)

        x = torch.cat([x, g, t], dim=-1)
        x = self.dense_layers(x)

        mu, sigma = torch.split(x, self.output_dim//2, dim=-1)
        sigma = torch.nn.functional.softplus(sigma) + 1e-5
        return self.max_action * torch.tanh(mu), sigma


def build_env(task):
    env_dict = {
        'PointReach': 'Point2DLargeEnv-v1',
        'PointRooms': 'Point2D-FourRoom-v1',
        'Reacher': 'Reacher-v2',
        'SawyerReach': 'SawyerReachXYZEnv-v1',
        'SawyerDoor': 'SawyerDoor-v0',
        'FetchReach': 'FetchReach-v2',
        'FetchPush': 'FetchPush-v2',
        'FetchPick': 'FetchPickAndPlace-v2',
        'FetchSlide': 'FetchSlide-v2',
        'HandReach': 'HandReach-v1',
    }
    env_id = env_dict[task]
    env = gym.make(env_id)
    if env_id.startswith('Fetch'):
        from envs.multi_world_wrapper import FetchGoalWrapper
        env._max_episode_steps = 50
        env = FetchGoalWrapper(env)
    elif env_id.startswith('Hand'):
        env._max_episode_steps = 50
    elif env_id.startswith('Sawyer'):
        from envs.multi_world_wrapper import SawyerGoalWrapper
        env = SawyerGoalWrapper(env)
        if not hasattr(env, '_max_episode_steps'):
            env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
    elif env_id.startswith('Point'):
        from envs.multi_world_wrapper import PointGoalWrapper
        env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
        env = PointGoalWrapper(env)
    elif env_id.startswith('Reacher'):
        from envs.multi_world_wrapper import ReacherGoalWrapper
        env._max_episode_steps = 50
        env = ReacherGoalWrapper(env)
    else:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
    return env

def modify_state_to_goal(env_name, states, goal_dim):
    if env_name.startswith('Point'):
        goal = states
    elif env_name.startswith('SawyerReach'):
        goal = states
    elif env_name.startswith('SawyerDoor'):
        goal = states[:, -goal_dim:]
    elif env_name.startswith('Reacher'):
        goal = states[:, -goal_dim-1:-1]
    elif env_name.startswith('FetchReach'):
        goal = states[:, :goal_dim]
    elif env_name.startswith('Fetch'):
        goal = states[:, goal_dim:2*goal_dim]
    elif env_name.startswith('HandReach'):
        goal = states[:, -goal_dim:]
    return goal

def create_img_data(task, variant):
    assert task in ['PointReach', 'PointRooms', 'SawyerReach', 'SawyerDoor'],\
          'Task not supported for image observations'

    dataset_path = os.path.join('offline_data', variant, task, 'buffer.pkl')
    save_path = os.path.join('offline_data', variant, task, 'image_buffer.pkl')

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    states = data['o']
    actions = data['u']
    num_traj, path_length, _ = states.shape

    env = build_env(task)

    if task.startswith('Point'):
        env.env.show_goal = False
        observations = np.zeros((num_traj, path_length, 48, 48, 3), dtype=np.uint8)
        for i in range(num_traj):
            for j in range(path_length):
                env.env.set_position(states[i, j])
                observations[i, j] = env.env.get_image()
    elif task.startswith('Sawyer'):
        observations = np.zeros((num_traj, path_length, 84, 84, 3), dtype=np.uint8)
        for i in range(num_traj):
            env.reset()
            for j in range(path_length-1):
                observations[i, j] = env.env.get_image()
                env.step(actions[i, j])
            observations[i, path_length-1] = env.env.get_image()

    data['o'] = observations
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def setup_logging(base_dir, env_name, variant, seed, diffusion_rollout, diffusion_nonparam):
    log_path = os.path.join(base_dir, 'logs', env_name, variant, str(seed))
    checkpoint_path = os.path.join(base_dir, 'checkpoints', env_name, variant, str(seed))
    buffer_path = os.path.join(log_path, 'diffusion_buffers', env_name, variant, str(seed))
    if diffusion_nonparam:
        log_path = os.path.join(log_path, 'forward_diffusion_nonparam')
        checkpoint_path = os.path.join(checkpoint_path, 'forward_diffusion_nonparam')
        buffer_path = os.path.join(buffer_path, 'forward_diffusion_nonparam')
    else:
        log_path = os.path.join(log_path, 'forward_diffusion_model')
        checkpoint_path = os.path.join(checkpoint_path, 'forward_diffusion_model')
        buffer_path = os.path.join(buffer_path, 'forward_diffusion_model')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(buffer_path, exist_ok=True)
    return log_path, checkpoint_path, buffer_path


class OfflineMerlin:
    def __init__(self, 
        env,
        dataset_path,
        device,
        seed,
        max_timesteps,
        max_path_length,
        test_horizon,
        her_ratio,
        latent=False,
        diffusion_nonparam=False,
        diffusion_rollout=True,
        diffusion_ratio=0.5,
        stitching_radius=1-1e-4,
        image_obs=False
    ):
        self.env = env
        self.dataset_path = dataset_path
        self.action_space = env.action_space
        self.action_dim = self.action_space.shape[0]
        self.action_dtype = self.action_space.dtype
        self.goal_space = env.observation_space['desired_goal']
        self.goal_dim = self.goal_space.shape[0]
        self.goal_dtype = self.goal_space.dtype
        self.image_obs = image_obs
        self.stitching_radius = stitching_radius

        if diffusion_nonparam:
            assert diffusion_rollout, 'Diffusion nonparam only supported with rollout'

        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)

        states = data['o'][0,0]
        self.state_shape = states.shape
        self.state_dtype = states.dtype

        variant = dataset_path.split('/')[-3]
        env_name = dataset_path.split('/')[-2]
        if self.image_obs:
            self.log_dir, self.checkpoint_dir, self.buffer_dir = setup_logging('./logging/img_obs', \
                                                                                env_name, variant, seed,
                                                                                diffusion_rollout,
                                                                                diffusion_nonparam)
        else:
            self.log_dir, self.checkpoint_dir, self.buffer_dir = setup_logging('./logging/state_obs',
                                                                                env_name, variant, seed,
                                                                                diffusion_rollout,
                                                                                diffusion_nonparam)
        if diffusion_rollout:
            if diffusion_nonparam:
                self.log_dir = os.path.join(self.log_dir, 'forward_diffusion_nonparam')
                
            else:
                self.log_dir = os.path.join(self.log_dir, 'forward_diffusion_model')
            os.makedirs(self.log_dir, exist_ok=True)

        self.device = device
        self.max_timesteps = max_timesteps
        self.max_path_length = max_path_length
        self.test_horizon = test_horizon
        self.her_ratio = her_ratio
        self.diffusion_nonparam = diffusion_nonparam
        self.diffusion_rollout = diffusion_rollout
        self.diffusion_ratio = diffusion_ratio
        self.latent = latent
        if self.diffusion_nonparam:
            self.latent = True

        if self.latent:
            self.encoder = train_contrastive_encoder(self.state_shape, self.dataset_path, \
                                        self.checkpoint_dir, self.device, pixel_based=False)
        else:
            self.encoder = None

        if dataset_path.split('/')[-2] in ['FetchPick', 'FetchPush', 'FetchSlide', 'HandReach']:
            max_buffer_size = 40000
        else:
            max_buffer_size = 2000

        self.dynamics = None
        self.policy = Policy(
            self.state_shape, 
            self.goal_dim, 
            32, 256,
            2*self.action_dim,
            max_path_length,
            self.action_space.high[0]
        ).to(self.device)

        self.dataset_replay_buffer = ReplayBuffer(
            self.state_shape,
            self.state_dtype,
            self.action_dim,
            self.action_dtype, 
            self.goal_dim,
            self.goal_dtype,
            self.max_path_length, 
            self.her_ratio,
            buffer_size=max_buffer_size,
        )

        if self.diffusion_rollout:
            self.buffer_filename = os.path.join(self.buffer_dir, 'buffer')
            self.diffusion_replay_buffer = ReplayBuffer(
                self.state_shape,
                self.state_dtype,
                self.action_dim,
                self.action_dtype, 
                self.goal_dim,
                self.goal_dtype,
                self.max_path_length, 
                self.her_ratio,
                buffer_size=max_buffer_size
            )

        self.policy_updates_per_step = None
        self.optimizer = optim.Adam(self.policy.parameters(), lr=5e-4)
        self.batch_size = 512

        # create the normalizer
        if not self.image_obs:
            self.o_norm = normalizer(size=self.state_shape[0])
        self.g_norm = normalizer(size=self.goal_dim)

        self.setup_data_and_models()

    def preprocess_and_load_dataset(self):
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)

        states = data['o']
        actions = data['u']
        goals = data['g'][:, 0]
        achieved_goals = data['ag']

        n_trajectories = states.shape[0]
        for i in range(n_trajectories):
            self.dataset_replay_buffer.add_trajectory(states[i], actions[i], goals[i], achieved_goals[i])

        return

    def setup_data_and_models(self):
        self.preprocess_and_load_dataset()
        start = time.time()
        if self.diffusion_rollout:
            self.collect_data = True
            if self.diffusion_nonparam:
                try:
                    self.diffusion_replay_buffer.load(self.buffer_filename + '.npz')
                    print('Buffer loaded')
                    self.collect_data = False
                except:
                    print('No buffer found, collecting new data...')
                    self.tree = StateTree(self.dataset_replay_buffer._states, self.stitching_radius, \
                                          self.max_path_length, self.device, self.latent, self.encoder)
                    print('Tree built')
            else:
                self.reverse_dynamics = train_dynamics(self.state_shape, self.action_dim, self.dataset_path, \
                                                self.checkpoint_dir, self.device, val=False)
                self.reverse_bc = train_reverse_bc(self.state_shape, self.action_dim, float(self.action_space.high[0]), self.dataset_path, \
                                                self.checkpoint_dir, self.device)    
        print('Model setup time: {}'.format(time.time()-start))        

    def forward_model_noise(self):
        states = np.zeros((self.rollout_batch_size, self.max_path_length, *self.state_shape))
        actions = np.zeros((self.rollout_batch_size, self.max_path_length-1, self.action_dim))
        achieved_states = np.zeros((self.rollout_batch_size, self.max_path_length, self.goal_dim))

        next_state, goal = self.dataset_replay_buffer.sample_goals(self.rollout_batch_size)

        T = self.max_path_length 
        states[:, T-1] = next_state
        achieved_states[:, T-1] = goal

        h = None
        for t in range(1, T):
            action = self.reverse_bc.select_action(next_state)
            state, h = self.reverse_dynamics.step(T-t-1, next_state, action, h)
            
            achieved_state = modify_state_to_goal(self.env.unwrapped.spec.id, state, self.goal_dim)

            states[:, T-t-1] = state
            actions[:, T-t-1] = action
            achieved_states[:, T-t-1] = achieved_state

            next_state = state

        states = states[~np.isnan(states).any(axis=1).any(axis=-1)]
        actions = actions[~np.isnan(actions).any(axis=1).any(axis=-1)]
        achieved_states = achieved_states[~np.isnan(achieved_states).any(axis=1).any(axis=-1)]
        return states, actions, achieved_states, goal

    def forward_tree_noise(self):
        states = np.zeros((self.rollout_batch_size, self.max_path_length, *self.state_shape))
        actions = np.zeros((self.rollout_batch_size, self.max_path_length-1, self.action_dim))
        achieved_states = np.zeros((self.rollout_batch_size, self.max_path_length, self.goal_dim))

        next_state, goal = self.dataset_replay_buffer.sample_goals(self.rollout_batch_size)

        T = self.max_path_length 
        states[:, T-1] = next_state
        achieved_states[:, T-1] = goal

        for t in range(1, T):
            buffer_idx = self.tree.query(next_state)

            delete_idx = np.nonzero(buffer_idx % self.max_path_length == 0)[0]
            buffer_idx = np.delete(buffer_idx, delete_idx)
            if len(buffer_idx) == 0:
                states = np.zeros(0)
                break
            goal = np.delete(goal, delete_idx, axis=0)
            states = np.delete(states, delete_idx, axis=0)
            actions = np.delete(actions, delete_idx, axis=0)
            achieved_states = np.delete(achieved_states, delete_idx, axis=0)

            traj_idx = buffer_idx // self.max_path_length
            time_idx = buffer_idx % self.max_path_length

            state, action, next_state, achieved_state = self.dataset_replay_buffer.get_transition(traj_idx, time_idx)

            states[:, T-t-1] = state
            actions[:, T-t-1] = action
            achieved_states[:, T-t-1] = achieved_state

            next_state = state

        return states, actions, achieved_states, goal

    def buffer_sample(self, buffer, batch_size):
        data = []
        for sub_buffer, ratio in buffer:
            sub_batch_size = int(batch_size * ratio + 0.1)
            sub_data = sub_buffer.sample_batch(sub_batch_size)
            data.append(sub_data)
            
        output = []
        for item in zip(*data):
            sub_data = np.concatenate(item, axis=0)
            output.append(sub_data)
        return output

    def train_policy(self):
        # collect diffusion trajectories
        self.rollout_batch_size = 1024
        start = time.time()
        if self.diffusion_rollout:
            while self.collect_data and self.diffusion_replay_buffer.current_buffer_size < self.diffusion_replay_buffer.max_buffer_size:
                self.rollout_batch_size = min(self.rollout_batch_size, self.diffusion_replay_buffer.max_buffer_size - self.diffusion_replay_buffer.current_buffer_size)
                if self.diffusion_nonparam:
                    states, actions, achieved_goals, goal = self.forward_tree_noise()
                else:
                    states, actions, achieved_goals, goal = self.forward_model_noise()
                for i in range(states.shape[0]):
                    self.diffusion_replay_buffer.add_trajectory(states[i], actions[i], goal[i], achieved_goals[i])
                print('{}/{}'.format(self.diffusion_replay_buffer.current_buffer_size, self.diffusion_replay_buffer.max_buffer_size))
        print('Diffusion data collection time: {}'.format(time.time()-start))
        
        if self.diffusion_nonparam and self.collect_data:
            self.diffusion_replay_buffer.save(self.buffer_filename)
        
        # setup buffer
        buffer = [(self.dataset_replay_buffer, 1.)]
        if self.diffusion_rollout:
            buffer = [(self.dataset_replay_buffer, 1. - self.diffusion_ratio),\
                    (self.diffusion_replay_buffer, self.diffusion_ratio)]

        # train policy
        if self.policy_updates_per_step is None:
            self.policy_updates_per_step = (self.max_path_length) * 10

        training_steps = 0
        train_losses = []

        best_score = 0
        dis_returns, undis_returns, successess = [], [], []

        if os.path.exists(os.path.join(self.checkpoint_dir, 'best_policy.pt')):
            self.policy.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "best_policy.pt")))
            while training_steps < self.max_timesteps:
                training_steps += self.policy_updates_per_step
                for _ in range(self.policy_updates_per_step):
                    _states, _actions, _goals, _, _horizons, _ = self.buffer_sample(buffer, self.batch_size)
                    self.o_norm.update(_states)
                    self.g_norm.update(_goals)
                    self.o_norm.recompute_stats()
                    self.g_norm.recompute_stats()
                    print(f"{training_steps}", end="\r")
        else:
            while training_steps < self.max_timesteps:
                training_steps += self.policy_updates_per_step
                _losses = []

                self.policy.train()
                for _ in range(self.policy_updates_per_step):
                    _states, _actions, _goals, _, _horizons, _ = self.buffer_sample(buffer, self.batch_size)

                    if not self.image_obs:
                        self.o_norm.update(_states)
                        self.o_norm.recompute_stats()
                    
                    self.g_norm.update(_goals)                        
                    self.g_norm.recompute_stats()

                    if self.image_obs:
                        _states = np.transpose(_states, (0, 3, 1, 2)) / 255.
                    else:
                        _states = self.o_norm.normalize(_states)
                    _goals = self.g_norm.normalize(_goals)

                    _states = torch.from_numpy(np.array(_states, dtype=np.float32)).to(self.device)
                    _goals = torch.from_numpy(np.array(_goals, dtype=np.float32)).to(self.device)
                    _actions = torch.from_numpy(np.array(_actions, dtype=np.float32)).to(self.device)
                    _horizons = torch.from_numpy(np.array(_horizons, dtype=np.float32)).to(self.device)
        
                    mus, sigmas = self.policy(_states, _goals, _horizons)
                    dist = Independent(Normal(mus, sigmas), 1)
                    loss = -dist.log_prob(_actions).mean()

                    self.optimizer.zero_grad()
                    loss.backward()

                    # Perform gradient clipping
                    max_grad_norm = 1.0  # Set the maximum gradient norm threshold
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)

                    self.optimizer.step()
                    _losses.append(np.mean(np.array(loss.item())))

                train_losses.append(np.mean(np.array(_losses)))
                
                print("Timesteps: {}, Loss: {}".format(training_steps, np.mean(np.array(loss.item()))))
                if training_steps % 2000 == 0:
                    dis, undis, succ = self.eval_policy(100, training_steps)
                    dis_returns.append(dis)
                    undis_returns.append(undis)
                    successess.append(succ)
                    np.save(os.path.join(self.log_dir, '{}_{}_dis_returns.npy'.format(int(10*self.her_ratio), self.test_horizon)), np.array(dis_returns))
                    np.save(os.path.join(self.log_dir, '{}_{}_undis_returns.npy'.format(int(10*self.her_ratio), self.test_horizon)), np.array(undis_returns))
                    np.save(os.path.join(self.log_dir, '{}_{}_successes.npy'.format(int(10*self.her_ratio), self.test_horizon)), np.array(successess))

                    if dis > best_score:
                        torch.save(self.policy.state_dict(), os.path.join(self.checkpoint_dir, 'best_policy.pt'))
                        best_score = dis
            
        dis, undis, succ = self.eval_policy(100, training_steps)
        dis_returns.append(dis)
        undis_returns.append(undis)
        successess.append(succ)
        np.save(os.path.join(self.log_dir, '{}_{}_dis_returns.npy'.format(int(10*self.her_ratio), self.test_horizon)), np.array(dis_returns))
        np.save(os.path.join(self.log_dir, '{}_{}_undis_returns.npy'.format(int(10*self.her_ratio), self.test_horizon)), np.array(undis_returns))
        np.save(os.path.join(self.log_dir, '{}_{}_successes.npy'.format(int(10*self.her_ratio), self.test_horizon)), np.array(successess))

    def eval_policy(self, num_episodes, iterations, plot=False):
        dis_returns = []
        undis_returns = []
        successes = []
        self.policy.eval()
        for i in range(num_episodes):
            obs, info = self.env.reset()
            rewards = []
            states, actions, goals = [], [], []
            done = False
            for t in range(self.max_path_length):
                if self.image_obs:
                    state = env.env.get_image()
                    state = np.transpose(state, (2, 0, 1)) / 255.
                else:
                    state = obs['observation']
                    state = self.o_norm.normalize(state)
                goal = obs['desired_goal']
                goal = self.g_norm.normalize(goal)

                with torch.no_grad():
                    input = (
                        torch.from_numpy(np.array(state, dtype=np.float32)).to(self.device), 
                        torch.from_numpy(np.array(goal, dtype=np.float32)).to(self.device),
                        torch.from_numpy(np.array([self.test_horizon], dtype=np.float32)).to(self.device)
                    )
                    action, _ = self.policy(*input)
                states.append(obs['observation'])
                actions.append(action.cpu().numpy())
                goals.append(obs['desired_goal'])

                obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
                done = terminated or truncated
                success = info['is_success']
                rewards.append(reward)

            dis_return, undis_return = discounted_return(rewards, 0.98, reward_offset=True)
            dis_returns.append(dis_return)
            undis_returns.append(undis_return)
            successes.append(success)

            if plot:
                states.append(obs['observation'])
                states = np.array(states)
                actions = np.array(actions)
                goals = np.array(goals)
                fig, ax = plt.subplots(figsize=(6, 6))
                env.plot_trajectory(ax, states, actions, goals[0])
                fig.savefig('{}_{}.png'.format(env.unwrapped.spec.id, i))
            
        dis_return = np.mean(np.array(dis_returns))
        undis_return = np.mean(np.array(undis_returns))
        success = np.mean(np.array(successes))
        print("Iterations: {}, Dis Return: {}, Undis Return: {}, Success: {}".format(iterations, dis_return, undis_return, success))

        return dis_return, undis_return, success

def discounted_return(rewards, gamma, reward_offset=True):
    L = len(rewards)
    if type(rewards[0]) == np.ndarray and len(rewards[0]):
        rewards = np.array(rewards).T
    else:
        rewards = np.array(rewards).reshape(1, L)

    if reward_offset:
        rewards += 1   # positive offset

    discount_weights = np.power(gamma, np.arange(L)).reshape(1, -1)
    dis_return = (rewards * discount_weights).sum(axis=1)
    undis_return = rewards.sum(axis=1)
    return dis_return.mean(), undis_return.mean()


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    # env args
    parser.add_argument("--task", type=str, default='PointRooms')
    parser.add_argument("--variant", type=str, default='expert')
    parser.add_argument("--max-timesteps", type=int, default=5e5)
    parser.add_argument("--max-path-length", type=int, default=50)
    parser.add_argument("--test-horizon", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--her-ratio", type=float, default=0.2)
    parser.add_argument("--stitching-radius", type=float, default=0.9999)
    parser.add_argument("--diffusion-nonparam", action='store_true')
    parser.add_argument("--diffusion-rollout", action='store_true')
    parser.add_argument("--image-obs", action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    register_envs()

    # override with tuned hyperparameters, uncomment for tuning
    args.her_ratio = HER_RATIO_TUNED[args.task][args.variant]
    args.test_horizon = HORIZON_TUNED[args.task][args.variant]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = build_env(args.task)
    filename = 'image_buffer.pkl' if args.image_obs else 'buffer.pkl' 
    dataset_path = os.path.join('./offline_data', args.variant, args.task, filename)

    # if img dataset doesn't exist, create it
    if not os.path.exists(dataset_path):
        print('Creating image dataset...')
        create_img_data(args.task, args.variant)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = OfflineMerlin(env, dataset_path, device, args.seed, args.max_timesteps, 
                            args.max_path_length, args.test_horizon, args.her_ratio, 
                            diffusion_nonparam=args.diffusion_nonparam, 
                            diffusion_rollout=args.diffusion_rollout, 
                            stitching_radius=args.stitching_radius, 
                            image_obs=args.image_obs)
    model.train_policy()