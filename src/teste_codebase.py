from class_experiment import Experimento
from GeraPlots import main_pipeline
import torch
import gc

num_neurons = 32
env_id = 'CartPole-v1'
params = {
    'policy_kwargs': dict(net_arch=[num_neurons, num_neurons]),
    'timesteps': int(1e6),
    'directory': f'../data/groups/{env_id}_{num_neurons}',
    'net_init': 2
}

enviroment_list = ['CartPole-v1']

inits = [3]
for enviroment in enviroment_list:
    for init in inits:
        params['directory'] = f'data/groups/{enviroment}_{num_neurons}_{init}'
        for seed in range(2):
            params['seeds'] = [seed]
            Ensaio = None  # Inicializa como None
            try:
                Ensaio = Experimento(params)
                Ensaio.treinamento()
            finally:
                if Ensaio is not None:
                    if hasattr(Ensaio, 'train_env'):
                        Ensaio.train_env.close()
                    if hasattr(Ensaio, 'model'):
                        if hasattr(Ensaio.model, 'rollout_buffer'):
                            del Ensaio.model.rollout_buffer
                        del Ensaio.model
                    del Ensaio
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()            
        main_pipeline(params['directory'])