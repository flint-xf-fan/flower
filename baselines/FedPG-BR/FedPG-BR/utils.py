"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

import torch
import numpy as np
from torch.nn import DataParallel

from matplotlib import animation
import matplotlib.pyplot as plt

       
def rollout(env, net, val_max_steps, device):
    obs = env.reset()
    done = False
    ep_rew = []
    step = 0
    while not done and step < val_max_steps:
        step += 1
        action = net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = True)[0]
        obs, rew, done, _ = env.step(action)
        ep_rew.append(rew)
    return np.sum(ep_rew), len(ep_rew), ep_rew

def set_parameters(params, net):
    model_actor = get_inner_model(net)
    model_actor.load_state_dict({**model_actor.state_dict(), **params})
    
def get_parameters(net):
    return  get_inner_model(net).parameters()
        
def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=120)
    