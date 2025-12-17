from class_experiment import Experimento
from GeraPlots import main_pipeline

num_neurons = 32
env_id = 'CartPole-v1'
params = {
    'policy_kwargs': dict(net_arch=[num_neurons, num_neurons]),
    'timesteps': int(1e6),
    'directory': f'../data/groups/{env_id}_{num_neurons}',
    'net_init': 2
}

enviroment_list = [
    # 'Acrobot-v1',
    'BipedalWalker-v3',
    # 'CarRacing-v3',
    # 'CartPole-v1',
    # 'MountainCar-v0',
    # 'Pendulum-v1',
    # 'MountainCarContinuous-v0',
    # 'LunarLander-v3'
]


for enviroment in enviroment_list:
    for init in range(10):
        params['directory'] = f'data/groups/{enviroment}_{num_neurons}_{init}'
        for seed in range(20):
            params['seeds'] = [seed]
            params['env_id'] = enviroment
            params['net_init'] = init
            Ensaio = None  # Inicializa como None
            Ensaio = Experimento(params)
            Ensaio.treinamento()
        main_pipeline(params['directory'])