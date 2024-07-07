import numpy as np
import os
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class OfflineDataset(Dataset):
    def __init__(self, dataset_path, indices):
        self.dataset_path = dataset_path
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        num_traj, path_length, _ = data['o'][:, :-1, :].shape
        self.states = data['o'][:, :-1, :].reshape((num_traj * path_length, -1))
        self.states = np.array(self.states, dtype=np.float32)[indices]
        self.actions = data['u'].reshape((num_traj * path_length, -1))
        self.actions = np.array(self.actions, dtype=np.float32)[indices]
        self.next_states = data['o'][:, 1:, :].reshape((num_traj * path_length, -1))
        self.next_states = np.array(self.next_states, dtype=np.float32)[indices]
        self.delta_states = self.next_states - self.states

        # normalize states
        eps = 1e-6
        self.states_mean = np.mean(self.states, axis=0, keepdims=True)
        self.states_std = np.std(self.states, axis=0, keepdims=True)
        self.states = (self.states - self.states_mean) / (self.states_std + eps)
        self.action_mean = np.mean(self.actions, axis=0, keepdims=True)
        self.action_std = np.std(self.actions, axis=0, keepdims=True)
        self.actions = (self.actions - self.action_mean) / (self.action_std + eps)
        self.next_states_mean = np.mean(self.next_states, axis=0, keepdims=True)
        self.next_states_std = np.std(self.next_states, axis=0, keepdims=True)
        self.next_states = (self.next_states - self.next_states_mean) / (self.next_states_std + eps)
        self.delta_states_mean = np.mean(self.delta_states, axis=0, keepdims=True)
        self.delta_states_std = np.std(self.delta_states, axis=0, keepdims=True)
        self.delta_states = (self.delta_states - self.delta_states_mean) / (self.delta_states_std + eps)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx], self.delta_states[idx]


class OfflineSeqDataset(Dataset):
    def __init__(self, dataset_path, indices):
        self.dataset_path = dataset_path
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        self.states = data['o'][:, :-1, :]
        self.states = np.array(self.states, dtype=np.float32)[indices]
        self.actions = data['u']
        self.actions = np.array(self.actions, dtype=np.float32)[indices]
        self.next_states = data['o'][:, 1:, :]
        self.next_states = np.array(self.next_states, dtype=np.float32)[indices]
        self.delta_states = self.next_states - self.states

        eps = 1e-6
        self.states_mean = np.mean(self.states, axis=0, keepdims=True)
        self.states_std = np.std(self.states, axis=0, keepdims=True)
        self.states = (self.states - self.states_mean) / (self.states_std + eps)
        self.action_mean = np.mean(self.actions, axis=0, keepdims=True)
        self.action_std = np.std(self.actions, axis=0, keepdims=True)
        self.actions = (self.actions - self.action_mean) / (self.action_std + eps)
        self.next_states_mean = np.mean(self.next_states, axis=0, keepdims=True)
        self.next_states_std = np.std(self.next_states, axis=0, keepdims=True)
        self.next_states = (self.next_states - self.next_states_mean) / (self.next_states_std + eps)
        self.delta_states_mean = np.mean(self.delta_states, axis=0, keepdims=True)
        self.delta_states_std = np.std(self.delta_states, axis=0, keepdims=True)
        self.delta_states = (self.delta_states - self.delta_states_mean) / (self.delta_states_std + eps)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx], self.delta_states[idx]


class DynamicsModel(nn.Module):
    def __init__(self, state_shape, action_dim, hidden_size):
        super(DynamicsModel, self).__init__()
        if len(state_shape) == 1:
            state_dim = state_shape[0]
        self.network = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )
    
    def forward(self, s_next, a):
        x = torch.cat([s_next, a], dim=-1)
        x = self.network(x)
        return x

class GRUModel(nn.Module):
    def __init__(self, state_shape, action_dim, hidden_size):
        super(GRUModel, self).__init__()
        if len(state_shape) == 1:
            state_dim = state_shape[0]
        self.gru = torch.nn.GRU(state_dim+action_dim, hidden_size, num_layers=2, \
                                    bias=True, batch_first=True)
        self.final = nn.Linear(hidden_size, state_dim)

    def forward(self, s_next, a, h):
        x = torch.cat([s_next, a], dim=-1)
        x, h_final = self.gru(x, h)
        x = self.final(x)
        return x, h_final

class VAE(nn.Module):
	def __init__(self, state_shape, action_dim, latent_dim, max_action, device):
		super(VAE, self).__init__()
		if len(state_shape) == 1:
			state_dim = state_shape[0]
		self.e1 = nn.Linear(state_dim + action_dim, 256)
		self.e2 = nn.Linear(256, 256)

		self.mean = nn.Linear(256, latent_dim)
		self.log_std = nn.Linear(256, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 256)
		self.d2 = nn.Linear(256, 256)
		self.d3 = nn.Linear(256, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device

	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], dim=-1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)

		u = self.decode(state, z)

		return u, mean, std

	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device) * 1e-3

		a = F.relu(self.d1(torch.cat([state, z], dim=-1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
        


class ReverseDynamics:
    def __init__(self, dataset, state_shape, action_dim, device):
        self.dataset = dataset
        self.action_dim = action_dim
        self.device = device

        self.dynamics_model = GRUModel(state_shape, action_dim, 256).to(device)
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics_model.parameters(), \
                                                   lr=1e-3)

    def step(self, t, next_state, action, h):
        self.dynamics_model.eval()

        eps = 1e-6
        action = (action - self.dataset.action_mean[:,t])/(self.dataset.action_std[:,t] + eps)
        next_state = (next_state - self.dataset.next_states_mean[:,t])/(self.dataset.next_states_std[:,t] + eps)

        with torch.no_grad():
            next_state_torch = torch.FloatTensor(next_state).to(self.device)
            action_torch = torch.FloatTensor(action).to(self.device)
            next_state_torch = next_state_torch.unsqueeze(1)
            action_torch = action_torch.unsqueeze(1)
            delta_state, h_next = self.dynamics_model(next_state_torch, action_torch, h)
            delta_state = delta_state.squeeze(1)
        
        delta_state = delta_state.cpu().numpy()
        deltas_unnormalized = self.dataset.delta_states_std[:,t] * delta_state + self.dataset.delta_states_mean[:,t]
        next_state_unnormalized = self.dataset.next_states_std[:,t] * next_state + self.dataset.next_states_mean[:,t]
        state = next_state_unnormalized - deltas_unnormalized
        return state, h_next


    def train(self, train_loader):
        self.dynamics_model.train()
        train_loss = 0.0

        for states, actions, next_states, delta_states in train_loader:
            states = states.to(self.device)
            next_states, actions, delta_states = next_states.to(self.device), actions.to(self.device), delta_states.to(self.device)
            
            outputs, _ = self.dynamics_model(next_states, actions, h=None)
            loss = F.mse_loss(outputs, delta_states)

            self.dynamics_optimizer.zero_grad()
            loss.backward()
            self.dynamics_optimizer.step()
            train_loss += loss.item()

        return train_loss / len(train_loader)

    def validate(self, val_loader):
        self.dynamics_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for states, actions, next_states, delta_states  in val_loader:
                states = states.to(self.device)
                next_states, actions, delta_states = next_states.to(self.device), actions.to(self.device), delta_states.to(self.device)
                
                outputs, _ = self.dynamics_model(next_states, actions, h=None)
                loss = F.mse_loss(outputs, states)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)

    def save(self, checkpoint_dir):
        torch.save(self.dynamics_model.state_dict(), os.path.join(checkpoint_dir, "RBC_dynamics"))
        torch.save(self.dynamics_optimizer.state_dict(), os.path.join(checkpoint_dir, "RBC_dynamics_optimizer"))

    def load(self, checkpoint_dir):
        if not torch.cuda.is_available():
            self.dynamics_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "RBC_dynamics"), map_location=torch.device('cpu')))
            self.dynamics_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "RBC_dynamics_optimizer"), map_location=torch.device('cpu')))
        else:
            self.dynamics_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "RBC_dynamics")))
            self.dynamics_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "RBC_dynamics_optimizer")))


class ReverseBC:
    def __init__(self, dataset, state_shape, action_dim, max_action, device, entropy_weight=0.5):
        latent_dim = action_dim * 2
        self.dataset = dataset

        self.vae = VAE(state_shape, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)

        self.max_action = max_action
        self.action_dim = action_dim
        self.device = device

        self.entropy_weight = entropy_weight
    
    def select_action(self, state):
        self.vae.eval()

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = self.vae.decode(state)
        action = action.cpu().numpy()
        return action
    
    def train(self, train_loader):
        self.vae.train()
        train_loss = 0.0
        for _, actions, next_states, _ in train_loader:
            actions = actions.cpu().numpy()
            next_states = next_states.cpu().numpy()
            actions = self.dataset.action_std * actions + self.dataset.action_mean
            next_states = self.dataset.next_states_std * next_states + self.dataset.next_states_mean

            actions = torch.FloatTensor(actions).to(self.device) 
            next_states = torch.FloatTensor(next_states).to(self.device)

            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(next_states, actions)
            recon_loss = F.mse_loss(recon, actions)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + self.entropy_weight * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()
            train_loss += vae_loss.item()
    
        return train_loss / len(train_loader)
        
    def save(self, checkpoint_dir):
        torch.save(self.vae.state_dict(), os.path.join(checkpoint_dir, "RBC_vae"))
        torch.save(self.vae_optimizer.state_dict(), os.path.join(checkpoint_dir, "RBC_vae_optimizer"))

    def load(self, checkpoint_dir):
        if not torch.cuda.is_available():
            self.vae.load_state_dict(torch.load(os.path.join(checkpoint_dir, "RBC_vae"), map_location=torch.device('cpu')))
            self.vae_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "RBC_vae_optimizer"), map_location=torch.device('cpu')))
        else:
            self.vae.load_state_dict(torch.load(os.path.join(checkpoint_dir, "RBC_vae")))
            self.vae_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "RBC_vae_optimizer")))


def train_test_indices(dataset_path, split_ratio=0.8):
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    num_traj, path_length, _ = data['o'][:, :-1, :].shape
    total_samples = num_traj
    num_train_samples = int(total_samples * split_ratio)

    # Create a list of indices for shuffling
    indices = list(range(total_samples))
    random.shuffle(indices)

    # Split the shuffled indices into training and validation
    train_indices = indices[:num_train_samples]
    val_indices = indices[num_train_samples:]

    return np.array(train_indices), np.array(val_indices)


def train_dynamics(state_shape, action_shape, dataset_path, checkpoint_path, device, val=False):
    batch_size = 256
    if val:
        train_indices, val_indices = train_test_indices(dataset_path, split_ratio=0.8)
        val_dataset = OfflineSeqDataset(dataset_path, val_indices)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_indices, _ = train_test_indices(dataset_path, split_ratio=1.0)
    train_dataset = OfflineSeqDataset(dataset_path, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    reverse_dynamics = ReverseDynamics(train_dataset, state_shape, \
                                    action_shape, device)
    try:
        reverse_dynamics.load(checkpoint_path)
        print('Loaded reverse dynamics model from checkpoint')
    except:
        print('No reverse dynamics checkpoint found, training from scratch...')
        num_epochs = 100
        average_val_loss = 0
        for epoch in range(num_epochs):
            average_train_loss = reverse_dynamics.train(train_loader)
            if val:
                average_val_loss = reverse_dynamics.validate(val_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}] '
                f'Train Loss: {average_train_loss:.4f} '
                f'Val Loss: {average_val_loss:.4f} ')
        print('Saving reverse dynamics model...')
        reverse_dynamics.save(checkpoint_path)
    return reverse_dynamics


def train_reverse_bc(state_shape, action_shape, max_action, dataset_path, checkpoint_path, device):
    batch_size = 256
    train_indices, _ = train_test_indices(dataset_path, split_ratio=1.0)
    train_dataset = OfflineDataset(dataset_path, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    reverse_bc = ReverseBC(train_dataset, state_shape, action_shape, \
                           max_action, device)
    
    try:
        reverse_bc.load(checkpoint_path)
        print('Loaded reverse BC model from checkpoint')
    except:
        print('No reverse BC checkpoint found, training from scratch...')
        num_epochs = 100
        for epoch in range(num_epochs):
            average_train_loss = reverse_bc.train(train_loader)

            # Print epoch statistics
            print(f'Epoch [{epoch + 1}/{num_epochs}] '
                f'Train Loss: {average_train_loss:.4f} ')
        print('Saving reverse BC model...')
        reverse_bc.save(checkpoint_path)
    return reverse_bc
