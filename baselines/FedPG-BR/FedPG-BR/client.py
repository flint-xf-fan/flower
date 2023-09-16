"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""
import flwr as fl
import torch
import numpy as np
from gym.spaces import Discrete
from models import MlpPolicy, DiagonalGaussianMlpPolicy
from utils import rollout, set_parameters

import gym

def gen_client_fn(env_name, model_config, train_config):
    
    env = gym.make(env_name)
    device = train_config.device
    
    hidden_units = model_config.hidden_units
    activation = model_config.activation
    output_activation = model_config.output_activation
    
    # get observation dim
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, Discrete):
        n_acts = env.action_space.n
    else:
        n_acts = env.action_space.shape[0]
    
    hidden_sizes = list(eval(hidden_units))
    sizes = [obs_dim]+hidden_sizes+[n_acts] # make core of policy network
    
    # get policy net
    if isinstance(env.action_space, Discrete):
        net = MlpPolicy(sizes, activation, output_activation).to(device)
    else:
        net = DiagonalGaussianMlpPolicy(sizes, activation).to(device)
    
    def client_fn(cid: str) -> FlowerClient:
        return FlowerClient(
            cid,
            net,
            env,
            train_config,
            device,
        )
    
    return client_fn
    

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, env, train_config, device):
        self.cid = cid
        self.net = net
        self.env = env
        self.train_config = train_config
        self.device = device

    def get_parameters(self, config):
        pass

    def collect_experience_for_training(self, B, record = False):
        # make some empty lists for logging.
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        batch_log_prob = []     # for gradient computing

        # reset episode-specific variables
        obs = self.env.reset()  # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep
        
        # make two lists for recording the trajectory
        if record:
            batch_states = []
            batch_actions = []

        t = 1
        # collect experience by acting in the environment with current policy
        while True:
            # save trajectory
            if record:
                batch_states.append(obs)

            act, log_prob = self.net(torch.as_tensor(obs, dtype=torch.float32).to(self.device), sample = True)
           
            obs, rew, done, info = self.env.step(act)
                
            # timestep
            t = t + 1
            
            # save action_log_prob, reward
            batch_log_prob.append(log_prob)
            
            ep_rews.append(rew)
            
            # save trajectory
            if record:
                batch_actions.append(act)

            if done or len(ep_rews) >= self.train_config.max_epi_len:
                
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                
                # the weight for each logprob(a_t|s_T) is sum_t^T (gamma^(t'-t) * r_t')
                returns = []
                R = 0
                # simulate random-reware attacker if needed
                for r in ep_rews[::-1]:
                    R = r + self.train_config.gamma * R
                    returns.insert(0, R)            
                returns = torch.tensor(returns, dtype=torch.float32)
                
                # return whitening
                advantage = (returns - returns.mean()) / (returns.std() + 1e-20)
                batch_weights += advantage

                # end experience loop if we have enough of it
                if len(batch_lens) >= B:
                    break
                
                # reset episode-specific variables
                obs, done, ep_rews, t = self.env.reset(), False, [], 1


        # make torch tensor and restrict to batch_size
        weights = torch.as_tensor(batch_weights, dtype = torch.float32).to(self.device)
        logp = torch.stack(batch_log_prob)

        if record:
            return weights, logp, batch_rets, batch_lens, batch_states, batch_actions
        else:
            return weights, logp, batch_rets, batch_lens
        
    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        
        set_parameters(self.net, parameters)
        batch_size = config.Batch_Size
        
        # collect experience by acting in the environment with current policy
        weights, logp, batch_rets, batch_lens = self.collect_experience_for_training(batch_size)
        
        # calculate policy gradient loss
        batch_loss = -(logp * weights).mean()
        
        # take a single policy gradient update step
        self.net.zero_grad()
        batch_loss.backward()
        grad = [item.grad for item in self.parameters()]

        return grad, batch_size, {} # return gradient instead of paramters !!!!!

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        
        val_ret = 0.0
        val_len = 0.0
         
        for _ in range(self.train_config.val_size):
            epi_ret, epi_len, _ = rollout(self.env, self.net, self.train_config.val_max_steps, self.device)
            val_ret += epi_ret
            val_len += epi_len
        val_ret /= self.opts.val_size
        val_len /= self.opts.val_size
             
        return val_ret, self.train_config.val_size, {"env length": float(val_len)}