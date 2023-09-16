"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union, Callable

import flwr as fl
import torch
import numpy as np
import torch.optim as optim

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

import gym
from gym.spaces import Discrete
from models import MlpPolicy, DiagonalGaussianMlpPolicy
from utils import get_parameters, set_parameters

class FedPGBR(fl.server.strategy.Strategy): ## !!!!!!!!!!!!!! have not finished
    def __init__(
        self,
        env_name, 
        model_config, 
        train_config,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.env_name = env_name, 
        self.model_config = model_config, 
        self.train_config = train_config,
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        
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
            net_old = MlpPolicy(sizes, activation, output_activation).to(device)
        else:
            net = DiagonalGaussianMlpPolicy(sizes, activation).to(device)
            net_old = DiagonalGaussianMlpPolicy(sizes, activation).to(device)
        self.net = net
        self.env = env
        self.train_config = train_config
        self.device = device
        self.net = net
        self.net_old = net_old
        
        # figure out the optimizer
        self.optimizer = optim.Adam(get_parameters(self.net), lr = self.train_config.lr_model)
        

    def __repr__(self) -> str:
        return "FedPGBR"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        ndarrays = get_parameters(self.net)
        return fl.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)
        self.n_clients = n_clients
        half_clients = n_clients // 2
        batch_size_Bmin = self.train_config.batch_size_Bmin
        batch_size_Bmax = self.train_config.batch_size_Bmax
        batch_size = np.random.randint(batch_size_Bmin, batch_size_Bmax + 1)
        self.batch_size = batch_size
        standard_config = {'Batch_Size': batch_size} ## !!!!!!!! check if work? not sure
        higher_lr_config = {}
        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )
        return fit_configurations
    
    
#  !!!!!!!!!!!!!!!!!!!!!!! we implement our algorithm here
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        gradient = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ] ### check if correct? how to get gradient sent back by clients?
        
        # make the old policy as a copy of the current master node
        set_parameters(self.net_old, set_parameters(self.net))
        
        # aggregate all detected non-Byzantine gradients to get mu
        mu = []
        for idx,item in enumerate(get_parameters(self.net_old)):
            grad_item = []
            for i in range(self.n_clients):
                if True: # only aggregate non-Byzantine gradients
                    grad_item.append(gradient[i][idx])
            mu.append(torch.stack(grad_item).mean(0))
        
        # SCSG
        b = self.train_config.minibatch_size_b
        N_t = np.random.geometric(p= 1 - self.batch_size/(self.batch_size + b))
        
        for n in range(N_t):
            self.optimizer.zero_grad()
            
            # sample b trajectory using the latest policy (\theta_n) of master node
            weights, new_logp, batch_rets, batch_lens, batch_states, batch_actions = self.collect_experience_for_training(b,
                                                                                                                         record = True,)
                
                 
            # calculate gradient for the new policy (\theta_n)
            loss_new = -(new_logp * weights).mean()
            self.net.zero_grad()
            loss_new.backward()
            if mu:
                # get the old log_p with the old policy (\theta_0) but fixing the actions to be the same as the sampled trajectory
                old_logp = []
                for idx, obs in enumerate(batch_states):
                    # act in the environment with the fixed action
                    _, old_log_prob = self.net_old(torch.as_tensor(obs, dtype=torch.float32).to(self.device), 
                                                                 fixed_action = batch_actions[idx])
                    # store in the old_logp
                    old_logp.append(old_log_prob)
                old_logp = torch.stack(old_logp)

                # Finding the ratio (pi_theta / pi_theta__old):
                # print(old_logp, new_logp)
                ratios = torch.exp(old_logp.detach() - new_logp.detach())
                
                # calculate gradient for the old policy (\theta_0)
                loss_old = -(old_logp * weights * ratios).mean()
                self.net_old.zero_grad()
                loss_old.backward()
                grad_old = [item.grad for item in get_parameters(self.net_old)]   
            
                # early stop if ratio is not within [0.995, 1.005]
                if torch.abs(ratios.mean()) < 0.995 or torch.abs(ratios.mean()) > 1.005:
                    N_t = n
                    break

                # adjust and set the gradient for latest policy (\theta_n)
                for idx,item in enumerate(get_parameters(self.net)):
                    item.grad = item.grad - grad_old[idx] + mu[idx]  # if mu is None, use grad from master

            # take a gradient step
            self.optimizer.step()
                  
        parameters_aggregated = get_parameters(self.net) # set new parameters
        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

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
        
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        # Let's assume we won't perform the global model evaluation on the server side.
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients