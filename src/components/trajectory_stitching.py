import numpy as np
from sklearn.neighbors import BallTree
import torch

class StateTree(object):
    def __init__(self, states, threshold, max_path_length, device, latent=False, encoder=None):
        self.states = states
        self.threshold = threshold
        self.max_path_length = max_path_length
        self.device = device
        self.latent = latent
        self.encoder = encoder.encoder

        if len(states.shape) > 2:
            states = states.reshape(-1, states.shape[-1])

        if self.latent:
            eps = 1e-6
            self.state_mean = np.mean(states, axis=0, keepdims=True)
            self.state_std = np.std(states, axis=0, keepdims=True)
            states = (states - self.state_mean) / (self.state_std + eps)
            with torch.no_grad():
                states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
                self.states = self.encoder(states).cpu().numpy()
        
        self.tree = BallTree(self.states, metric='euclidean') # can try manhattan distance

    def query(self, state, k=1):
        buffer_idx = []
        switched_idx = []

        new_count = 0
        eps = 1e-6

        if self.latent:
            state = (state - self.state_mean) / (self.state_std + eps)
            with torch.no_grad():
                state = torch.from_numpy(np.array(state, dtype=np.float32)).to(self.device)
                state = self.encoder(state).cpu().numpy()

        _, index = self.tree.query(state, k=2, dualtree=False, sort_results=True)
        n_state = self.states[index[:,1]]
        cosine = np.sum(state * n_state, axis=1) / (np.linalg.norm(state, axis=1) * np.linalg.norm(n_state, axis=1))

        for i, ind  in enumerate(index):
            if cosine[i] >= self.threshold and ind[1] % self.max_path_length != 0:
                buffer_idx.append(ind[1])
                new_count += 1
                switched_idx.append(i)
            else:
                buffer_idx.append(ind[0])

        # print(new_count, switched_idx)
        buffer_idx = np.array(buffer_idx)
        
        return buffer_idx