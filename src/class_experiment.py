from stable_baselines3.common.monitor import Monitor
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import gymnasium
import torch 
import json
import os
import re

from class_ppo import PPO_tunado
plt.style.use('style.mplstyle')

def criar_pasta(directory: str) -> str:
    max_number = 0
    pattern = re.compile(r'^(\d{3})$')

    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)) and pattern.match(item):
            try:
                current_number = int(item)
                if current_number > max_number:
                    max_number = current_number
            except ValueError:
                pass

    next_number = max_number + 1
    new_folder_name = str(next_number).zfill(3)
    new_directory = os.path.join(directory, new_folder_name)

    os.makedirs(new_directory)
    return new_directory
    
def gera_combinacoes(col_info: dict) -> list[tuple[str, str]]:
    combinacoes_sequenciais = []
    for col1, info1 in col_info.items():
        if len(info1) == 2:  # Certifica-se de que a coluna tem dois elementos (para ter um "segundo")
            segundo_elemento_col1 = info1[1]
            for col2, info2 in col_info.items():
                if col1 != col2 and len(info2) == 2:  # Evita comparar a mesma coluna e garante dois elementos
                    primeiro_elemento_col2 = info2[0]
                    if segundo_elemento_col1 == primeiro_elemento_col2:
                        combinacoes_sequenciais.append((col1, col2))
    return combinacoes_sequenciais

def combinar_strings(tupla_de_strings: tuple[str, str]) -> str:
    """
    Combina duas strings do tipo I(a,b) e I(b,c) em I(a,b,c).
    """
    primeira = tupla_de_strings[0]
    segunda = tupla_de_strings[1]
    
    args_primeira_str = re.search(r'I\((.*)\)', primeira).group(1) #type:ignore
    args_segunda_str = re.search(r'I\((.*)\)', segunda).group(1) #type:ignore
    
    lista_args = args_primeira_str.split(',') + args_segunda_str.split(',')
    args_combinados_unicos = list(dict.fromkeys(lista_args))
    
    return f'I({",".join(args_combinados_unicos)})'


def fechar_plot(directory, plot_name, axle_x = 'Norma', axle_y = 'Época'):
    plt.xlabel(axle_x)
    plt.ylabel(axle_y)
    if not plt.gca().get_legend_handles_labels():
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'{directory}/plots/{plot_name}.pdf')
    plt.close()

class Experimento():
    def __init__(self, params) -> None:
        # Setagem
        self.env_id = 'CartPole-v1'
        self.n_envs = 4
        self.policy_kwargs = dict(net_arch = [32,32])
        self.seeds = [0]
        self.timesteps = int(1e3) 
        self.reference_agent = None
        self.calc_mutual_info = True
        self.directory = '../data/results'

        # Model Parameters
        self.learning_rate = 3e-4
        self.n_steps = 2048
        self.batch_size = 64
        self.n_epochs = 10
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.clip_range_vf = None
        self.normalize_advantage = True
        self.ent_coef = 0.0
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.use_sde = False
        self.sde_sample_freq = -1
        self.rollout_buffer_class = None
        self.rollout_buffer_kwargs = None
        self.target_kl = None
        self.stats_window_size = 100
        self.tensorboard_log =  "../data/tensorboard_logs/"
        self.verbose = 0
        self.seed = 0
        self.device = "cpu"
        
        # Recording Parameters
        self.recording = False
        self.recording_ep_freq = 100
        self.coleta = False
        # Fim da setagem

        for chave, valor in params.items():
            setattr(self, chave, valor)
        
        #criar uma pasta com número maior sem perder a ordenação
        self.directory = criar_pasta(self.directory)

        self._hyperparams = {
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'clip_range_vf': self.clip_range_vf,
            'normalize_advantage': self.normalize_advantage,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'use_sde': self.use_sde,
            'sde_sample_freq': self.sde_sample_freq,
            'rollout_buffer_class': self.rollout_buffer_class,
            'rollout_buffer_kwargs': self.rollout_buffer_kwargs,
            'target_kl': self.target_kl,
            'stats_window_size': self.stats_window_size,
            'tensorboard_log': self.tensorboard_log,
            'policy_kwargs': self.policy_kwargs,
            'verbose': self.verbose,
            'device': self.device
        }        

        self.train_env = Monitor(gymnasium.make(self.env_id))
        # self.train_env.reset(seed = self.seed)

        #* Reprodutibilidade
        torch.manual_seed(1)
        
        self.model = PPO_tunado(self.directory, 'MlpPolicy', self.train_env, self.reference_agent, self.calc_mutual_info, self._hyperparams)
    
    def plots(self):
        os.makedirs(f'{self.directory}/plots')
        data = pd.read_csv(f'{self.directory}/resultados.csv')
        col_info = {col: [item.strip() for item in col.strip('I()').split(',')] for col in data.columns}
        combinacoes_sequenciais = gera_combinacoes(col_info=col_info)

        # loss
        loss_data = data.filter(like= 'loss')
        loss_data.plot()
        fechar_plot(self.directory, 'loss')
        

        # recompensa
        data_to_plot = pd.read_csv(f'{self.directory}/rewards.csv')
        rewards = data_to_plot.T
        means = rewards.apply(np.mean)
        stds = rewards.apply(np.std)
        plt.plot(means, label='Média', color='blue', marker='o')
        plt.fill_between(
            range(len(means)),          # Eixo x (iterações)
            np.array(means) - np.array(stds),  # Limite inferior
            np.array(means) + np.array(stds),  # Limite superior
            alpha=0.2,                 # Transparência
            color='blue',
            label='±1 Desvio Padrão'
        )
        fechar_plot(self.directory, 'reward', 'Iteração', 'Recompensa')
                
        
        # ator - pesos e gradiente
        data.filter(like = 'actor_weight').plot()
        fechar_plot(self.directory, 'actor_weight')


        data.filter(like = 'actor_grad').plot()
        fechar_plot(self.directory, 'actor_grad')

        # critico - pesos e gradiente
        data.filter(like = 'critic_weight').plot()
        fechar_plot(self.directory, 'critic_weight')
        

        data.filter(like = 'critic_grad').plot()
        fechar_plot(self.directory, 'critic_grad')

        # informação mútua
        size = len(data[combinacoes_sequenciais[0][0]])
        for col1, col2 in combinacoes_sequenciais:
            val1 = data[col1] #data.groupby(data.index // self.n_steps)[col1].mean()
            val2 = data[col2] #data.groupby(data.index // self.n_steps)[col2].mean()
            plt.scatter(val1, val2, c= np.arange(0, len(data[col1])), cmap= 'magma')
            col_x_name = col1.strip('').replace('hat', '\\hat')
            col_y_name = col2.strip('').replace('hat', '\\hat')
            title = f'{combinar_strings((col1, col2))}'
            fechar_plot(self.directory, title, f'${col_x_name}$', f'${col_y_name}$')
        
        # Colorbar
        fig, ax = plt.subplots(figsize=(12, .5)) # Ajuste o tamanho para ser mais largo e fino
    
        norm = Normalize(vmin=0, vmax= size)
        cmap = plt.get_cmap('magma')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) # Adiciona um array vazio para o ScalarMappable
        
        cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
        cbar.set_label('Épocas') # Ajuste a legenda conforme seus dados
        plt.savefig(f'{self.directory}/plots/colorbar.pdf')
        plt.close()
        
    def treinamento(self):
        for seed in tqdm(self.seeds, desc="Training with different seeds"):
            self.model.set_random_seed(seed)
            temp2 = self.model.learn(total_timesteps= self.timesteps, progress_bar= False)
            temp = 2
        
        # recompensa
        df = pd.DataFrame(self.model.rewards_list)
        rewards_directory = os.path.join(self.directory, 'rewards.csv')
        df.to_csv(rewards_directory, mode= 'w', index=False, header= True)    

        self.model.save(os.path.join(self.directory, 'agente_treinado'))

        params = {chave: valor for chave, valor in self.__dict__.items() if not chave in ['train_env', 'model', '_hyperparams']}
        json_string = json.dumps(params, indent= 4)
        json_path = os.path.join(self.directory, f'{self.env_id}-{self.timesteps}.json')
        with open(json_path, 'w') as arquivo:      
            arquivo.write(json_string)

        self.plots()
        self.train_env.close()