
import numpy as np
import os
import math
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
        self.next_states = data['o'][:, 1:, :].reshape((num_traj * path_length, -1))
        self.next_states = np.array(self.next_states, dtype=np.float32)[indices]

        # normalize states
        eps = 1e-6
        self.state_mean = np.mean(self.states, axis=0, keepdims=True)
        self.state_std = np.std(self.states, axis=0, keepdims=True)
        self.states = (self.states - self.state_mean) / (self.state_std + eps)
        self.next_states_mean = np.mean(self.next_states, axis=0, keepdims=True)
        self.next_states_std = np.std(self.next_states, axis=0, keepdims=True)
        self.next_states = (self.next_states - self.next_states_mean) / (self.next_states_std + eps)


    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.next_states[idx]


class MLPEncoder(nn.Module):
    def __init__(self, state_dim, hidden_size=512, embedding_dim=16):
        super(MLPEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, embedding_dim)
        )
        self.output_dim = embedding_dim
    
    def forward(self, x):
        return self.network(x)


class CNNEncoder(nn.Module):
    def __init__(self, c, h, w, embedding_dim=16):
        super(CNNEncoder, self).__init__()
        # Convolutional layers
        self.network = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.output_dim = int(np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:]))
        self.net = nn.Sequential(
            self.net,
            nn.Linear(self.output_dim, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = embedding_dim

    def forward(self, x):
        return self.network(x)


class ContrastiveRepresentationLearner:
    def __init__(self, state_shape, device, temperature=0.5, pixel_based=False):
        self.device = device
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

        if pixel_based:
            c, h, w = state_shape
            self.encoder = CNNEncoder(c, h, w).to(device)
        else:
            state_dim = state_shape[0]
            self.encoder = MLPEncoder(state_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=3e-4)
        self.output_dim = self.encoder.output_dim

    def info_nce_loss(self, state_rep, next_state_rep):
        eps = 1e-6

        labels = torch.arange(state_rep.shape[0])
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        state_rep = F.normalize(state_rep, dim=1)
        next_state_rep = F.normalize(next_state_rep, dim=1)

        cov = torch.mm(state_rep, next_state_rep.T) # (b,b)
        sim = torch.exp(cov / self.temperature) 
        neg = sim.sum(dim=-1) # (b,)
        row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / self.temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = torch.exp(torch.sum(state_rep * next_state_rep, dim=-1) / self.temperature) #(b,)
        loss = -torch.log(pos / (neg + eps)) #(b,)
        loss = loss.mean()

        return loss

    def train(self, train_loader):
        self.encoder.train()
        train_loss = 0.0

        for states, next_states in train_loader:
            states, next_states = states.to(self.device), next_states.to(self.device)
            states_rep = self.encoder(states)
            next_states_rep = self.encoder(next_states)
            loss = self.info_nce_loss(states_rep, next_states_rep)

            self.encoder_optimizer.zero_grad()
            loss.backward()
            self.encoder_optimizer.step()
            train_loss += loss.item()

        return train_loss / len(train_loader)

    def save(self, checkpoint_dir):
        torch.save(self.encoder.state_dict(), os.path.join(checkpoint_dir, "contrastive_encoder"))
        torch.save(self.encoder_optimizer.state_dict(), os.path.join(checkpoint_dir, "contrastive_encoder_optimizer"))

    def load(self, checkpoint_dir):
        if not torch.cuda.is_available():
            self.encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "contrastive_encoder"), map_location=torch.device('cpu')))
            self.encoder_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "contrastive_encoder_optimizer"), map_location=torch.device('cpu')))
        else:
            self.encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "contrastive_encoder")))
            self.encoder_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "contrastive_encoder_optimizer")))

def train_test_indices(dataset_path, split_ratio=0.8):
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    num_traj, path_length, _ = data['o'][:, :-1, :].shape
    total_samples = num_traj * path_length # -1 because we need a next state
    num_train_samples = int(total_samples * split_ratio)

    # Create a list of indices for shuffling
    indices = list(range(total_samples))
    random.shuffle(indices)

    # Split the shuffled indices into training and validation
    train_indices = indices[:num_train_samples]
    val_indices = indices[num_train_samples:]

    return np.array(train_indices), np.array(val_indices)

def train_contrastive_encoder(state_shape, dataset_path, checkpoint_path, device, pixel_based=False):
    batch_size = 256
    train_indices, _ = train_test_indices(dataset_path, split_ratio=1.0)
    train_dataset = OfflineDataset(dataset_path, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    encoder = ContrastiveRepresentationLearner(state_shape, \
                                 device, pixel_based=pixel_based)
    try:
        encoder.load(checkpoint_path)
        print('Loaded contrastive encoder from checkpoint')
    except:
        print('No contrastive encoder checkpoint found, training from scratch...')
        num_epochs = 100
        average_val_loss = 0
        for epoch in range(num_epochs):
            average_train_loss = encoder.train(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}] '
                f'Train Loss: {average_train_loss:.4f} '
                f'Val Loss: {average_val_loss:.4f} ')
        print('Saving contrastive encoder...')
        encoder.save(checkpoint_path)
    return encoder