from src.class_experiment import Experimento
from src.post_processing import main_pipeline

num_neurons = 32
env_id = "CartPole-v1"
params = {
    "policy_kwargs": dict(net_arch=[num_neurons, num_neurons]),
    "timesteps": int(1e4),
    "directory": f"data/groups/{env_id}_{num_neurons}",
    "net_init": 2,
}

environment_list = [
    # 'Acrobot-v1',
    "BipedalWalker-v3",
    # 'CarRacing-v3',
    # 'CartPole-v1',
    # 'MountainCar-v0',
    # 'Pendulum-v1',
    # 'MountainCarContinuous-v0',
    # 'LunarLander-v3'
]


for environment in environment_list:
    for init in range(2):
        # todo melhorar essa nomeação
        params["directory"] = "data/test/A0"
        for seed in range(2):
            params["seeds"] = [seed]
            params["env_id"] = environment
            params["net_init"] = init
            ensaio = None  # Inicializa como None
            ensaio = Experimento(params)
            ensaio.treinamento()
        main_pipeline(params["directory"])
