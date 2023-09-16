"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import hydra
import flwr as fl
from omegaconf import DictConfig, OmegaConf
from client import gen_client_fn
from strategy import FedPGBR
from hydra.utils import instantiate

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print('run FedScsPG with the following config: \n')
    print(OmegaConf.to_yaml(cfg))
    
    dataset_config = cfg.dataset
    model_config = cfg.model
    train_config = cfg.train
    client_config = cfg.client

    # 2. Prepare your dataset
    # here you should call a function in datasets.py that returns whatever is needed to:
    # (1) ensure the server can access the dataset used to evaluate your model after
    # aggregation
    # (2) tell each client what dataset partitions they should use (e.g. a this could
    # be a location in the file system, a list of dataloader, a list of ids to extract
    # from a dataset, it's up to you)
    env_name = dataset_config.env # No need to prepare dataset as we are based on RL
    print('env name defined')

    # 3. Define your clients
    # Define a function that returns another function that will be used during
    # simulation to instantiate each individual client
    # client_fn = client.<my_function_that_returns_a_function>()
    client_fn = gen_client_fn(
        env_name, 
        model_config, 
        train_config,)
    print('client defined')
    
    # 4. Define your strategy
    # pass all relevant argument (including the global dataset used after aggregation,
    # if needed by your method.)
    # strategy = instantiate(cfg.strategy, <additional arguments if desired>)
    strategy = FedPGBR(
        env_name, 
        model_config, 
        train_config,)
    print('strategy defined')

    # 5. Start Simulation
    print('start simulation')
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=client_config.num_worker,
        config=fl.server.ServerConfig(num_rounds=100), # !!!!!! number of runs is not max number of traj
        client_resources={
            "num_cpus": client_config.num_cpus,
        },
        strategy=strategy,
    )

    # 6. Save your results
    # Here you can save the `history` returned by the simulation and include
    # also other buffers, statistics, info needed to be saved in order to later
    # on generate the plots you provide in the README.md. You can for instance
    # access elements that belong to the strategy for example:
    # data = strategy.get_my_custom_data() -- assuming you have such method defined.
    # Hydra will generate for you a directory each time you run the code. You
    # can retrieve the path to that directory with this:
    # save_path = HydraConfig.get().runtime.output_dir
    print("................")
    print(history) # to do !!!!!!!!!!!!!!!!!!!
    
if __name__ == "__main__":
    main()
