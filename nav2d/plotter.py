import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

import torch
from torch.distributions import Normal, Independent

ROOT_DIR = './plots/'


def plot_mesh(env, policy, device, name='plot', test_horizon=20, ood=False, gcsl=False):
    os.makedirs(ROOT_DIR, exist_ok=True)
    x_min = env.observation_space.low[0]
    x_max = env.observation_space.high[0]
    y_min = env.observation_space.low[1]
    y_max = env.observation_space.high[1]
    x_range = np.linspace(x_min, x_max, num=40)  # Adjust as needed
    y_range = np.linspace(y_min, y_max, num=40)  # Adjust as needed

    # Generate the meshgrid
    X, Y = np.meshgrid(x_range, y_range)
    all_states = np.array(np.vstack([X.ravel(), Y.ravel()]).T, dtype=np.float32)
    all_states = torch.from_numpy(all_states).to(device)
    horizon = torch.tensor([test_horizon]).expand(all_states.shape[0], -1).to(device)

    if gcsl:
        horizon = 0

    _, info = env.reset()
    goals = [info['goal']]

    # for ood goal
    if ood:
        goals = [
        np.array([5.,4.], dtype=np.float32),
        np.array([7.,-8.], dtype=np.float32),
        np.array([-3.,-6.], dtype=np.float32),
        np.array([-7.,4.], dtype=np.float32),
    ]

    for i, goal in enumerate(goals):
        goal = torch.from_numpy(goal).expand(all_states.shape[0], -1).to(device) 

        with torch.no_grad():
            if gcsl:
                actions = policy(all_states, goal).cpu().numpy()
            else:
                mus, kappas = policy(all_states, goal, horizon)
                dist = Independent(Normal(mus, kappas), 1)
                actions = dist.sample().cpu().numpy()
            
        goal = goal[0].cpu().numpy()

        actions = actions / np.linalg.norm(actions, axis=-1, keepdims=True)
        delta_x = actions[:,0]
        delta_y = actions[:,1]
        delta_x = delta_x.reshape(X.shape)
        delta_y = delta_y.reshape(Y.shape)

        if gcsl:
            title_str = 'Mean actions from GCSL policy'
        else:
            title_str = 'Sampled actions from diffusion policy'

        plt.figure(figsize=(8, 8))
        plt.quiver(X, Y, delta_x, delta_y, angles='xy', scale=6, scale_units='inches', color='k')
        plt.plot(env.goal[0], env.goal[1], marker='x', color='r', markersize=15, mew=5)
        if ood:
            plt.plot(goal[0], goal[1], marker='x', color='g', markersize=15, mew=5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title_str)
        plt.savefig(ROOT_DIR + '{}_{}.pdf'.format(name, i))
        plt.close()

def plot_noisy_trajectories(env, states, max_path_length, name='diff_trajectories'):
    points = np.flip(np.array(states), axis=1)
    cmap = plt.get_cmap('viridis')

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot each trajectory as a continuous line with changing colors
    for i in range(len(points)):
        num_points = len(points[i])
        x, y = zip(*points[i])  # Unzip the (x, y) points from the trajectory
        colors = np.linspace(0, 1, num_points)  # Create a linear colormap from 0 to 1
        line_segments = np.array([[[x[i], y[i]], [x[i + 1], y[i + 1]]] for i in range(num_points - 1)])
        cmap_colors = cmap(colors)  # Map the linear colormap to colors
        norm = plt.Normalize(0, 1)
        lc = LineCollection(line_segments, cmap=LinearSegmentedColormap.from_list('custom_cmap', cmap_colors), norm=norm)
        lc.set_array(colors)
        ax.add_collection(lc)

    ax.plot(env.goal[0], env.goal[1], marker='x', color='r', markersize=15, mew=5)
    plt.title("Noisy Trajectories")
    plt.xlim(env.observation_space.low[0], env.observation_space.high[0])
    plt.ylim(env.observation_space.low[1], env.observation_space.high[1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(ROOT_DIR + '{}.pdf'.format(name))
    plt.close()