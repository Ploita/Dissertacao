from stable_baselines3.common.monitor import Monitor
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import gymnasium
import keyboard
import torch
import json
import os

from utils import CustomCallback, fib
from class_ppo import PPO_tunado
plt.style.use('style.mplstyle')

class Experimento():
    def __init__(self, params) -> None:
        # Setagem
        self.env_id = 'CartPole-v1'
        self.n_envs = 4
        self.policy_kwargs = dict(net_arch = [32,32])
        self.fib_seeds = fib(5)
        self.timesteps = int(1e3) 
        self.reference_agent = None
        self.controller = None
        self.calc_mutual_info = False
        self.reference_control = False

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
        self.tensorboard_log =  "./tensorboard_logs/"
        self.policy_kwargs = None
        self.verbose = 0
        self.seed = None
        self.device = "cpu"
        
        # Recording Parameters
        self.recording = False
        self.recording_ep_freq = 100
        self.device = 'auto'
        self.coleta = False
        # Fim da setagem

        for chave, valor in params.items():
            setattr(self, chave, valor)
        
        #isso serve pra criar uma pasta com número maior sem perder a ordenação
        if not os.path.exists(f'{self.directory}/000'):
            os.makedirs(f'{self.directory}/000')
            self.directory = f'{self.directory}/000'
        else:
            last_number = os.listdir(self.directory)[-1]

            self.directory = os.path.join(self.directory, str(int(last_number) + 1).zfill(3))
            os.makedirs(self.directory)

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
            'seed': self.seed,
            'device': self.device
        }        

        self.train_env = Monitor(gymnasium.make(self.env_id))
        if self.seed is not None:
            self.train_env.reset(seed = self.seed[0])

        if self.reference_control:
            self.controller = Controller(self.env.get_attr('env')[0]) #type: ignore

        #* Reprodutibilidade
        torch.manual_seed(0)
        
        self.model = PPO_tunado(self.directory, 'MlpPolicy', self.train_env, self.reference_agent, self.controller, self.calc_mutual_info, self._hyperparams) 
    
    def plots(self):
        os.makedirs(f'{self.directory}/plots')
        data = pd.read_csv(f'{self.directory}/resultados.csv')
        col_info = {col: [item.strip() for item in col.strip('train/I()').split(',')] for col in data.columns}
        combinacoes_sequenciais = []

        # loss + recompensa
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        loss_data = data.filter(like= 'loss')
        loss_data.plot(title= 'Loss', ax=axs[0])  
        axs[0].set_ylabel('Norma')
        axs[0].set_xlabel('Época')
        axs[0].legend()
        axs[0].text(0.02,0.98,'a)', transform=axs[0].transAxes, va='top')

        data_to_plot = pd.read_csv(f'{self.directory}/rewards.csv')
        rewards = data_to_plot.T
        means = rewards.apply(np.mean)
        stds = rewards.apply(np.std)
        axs[1].plot(means, label='Média', color='blue', marker='o')
        axs[1].fill_between(
            range(len(means)),          # Eixo x (iterações)
            np.array(means) - np.array(stds),  # Limite inferior
            np.array(means) + np.array(stds),  # Limite superior
            alpha=0.2,                 # Transparência
            color='blue',
            label='±1 Desvio Padrão'
        )
        axs[1].set_xlabel('Iteração')
        axs[1].set_ylabel('Recompensa')
        axs[1].set_title('Distribuição das Recompensas por Iteração com Faixa de Desvio Padrão')
        axs[1].legend()
        axs[1].text(0.02,0.98,'b)', transform=axs[1].transAxes, va='top')
        plt.tight_layout()
        plt.savefig(f'{self.directory}/plots/loss+reward.pdf')
        plt.close()

                
        # pesos e gradiente
        # ator
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        data.filter(like = 'train/actor_weight').plot(ax=axs[0])
        axs[0].set_xlabel('Época')
        axs[0].set_ylabel('Norma')
        axs[0].set_title('Evolução dos pesos')
        axs[0].legend(loc = 'upper right')
        axs[0].text(0.02,0.98,'a)', transform=axs[0].transAxes, va='top')


        data.filter(like = 'train/actor_grad').plot(ax=axs[1])
        axs[1].set_xlabel('Época')
        axs[1].set_ylabel('Norma')
        axs[1].set_title('Evolução do gradiente')
        axs[1].legend(loc = 'upper right')
        axs[1].text(0.02,0.98,'b)', transform=axs[1].transAxes, va='top')
        plt.suptitle('Rede Ator')
        plt.tight_layout()
        plt.savefig(f'{self.directory}/plots/actor.pdf')
        plt.close()

        # ## critico
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        data.filter(like = 'train/critic_weight').plot(ax=axs[0])
        axs[0].set_xlabel('Época')
        axs[0].set_ylabel('Norma')
        axs[0].set_title('Evolução dos pesos')
        axs[0].legend(loc = 'upper center')
        axs[0].text(0.02,0.98,'a)', transform=axs[0].transAxes, va='top')


        data.filter(like = 'train/critic_grad').plot(ax=axs[1])
        axs[1].set_xlabel('Época')
        axs[1].set_ylabel('Norma')
        axs[1].set_title('Evolução do gradiente')
        axs[1].legend()
        axs[1].text(0.02,0.98,'b)', transform=axs[1].transAxes, va='top')
        plt.suptitle('Rede Crítico')
        plt.tight_layout()
        plt.savefig(f'{self.directory}/plots/critic.pdf')
        plt.close()

        # informação mútua
        for col1, info1 in col_info.items():
            if len(info1) == 2:  # Certifica-se de que a coluna tem dois elementos (para ter um "segundo")
                segundo_elemento_col1 = info1[1]
                for col2, info2 in col_info.items():
                    if col1 != col2 and len(info2) == 2:  # Evita comparar a mesma coluna e garante dois elementos
                        primeiro_elemento_col2 = info2[0]
                        if segundo_elemento_col1 == primeiro_elemento_col2:
                            combinacoes_sequenciais.append((col1, col2))
        size = len(data[combinacoes_sequenciais[0][0]])
        norm = Normalize(vmin=0, vmax= size)
        
        if len(combinacoes_sequenciais) == 4:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            axs = axs.ravel()  # Transforma em array 1D para facilitar iteração
            subplot_labels = ['a)', 'b)', 'c)', 'd)']
            for i in range(4):
                col1 = combinacoes_sequenciais[i][0]
                col2 = combinacoes_sequenciais[i][1]
                axs[i].scatter(data[col1], data[col2],c= np.arange(0, size),cmap= 'magma',norm=norm)
                col1 = col1.strip('train/')
                col2 = col2.strip('train/').replace('hat', '\\hat')
                title = f'Relação ${col1}\\times {col2}$'
                axs[i].set_title(title)
                axs[i].set_xlabel(f'${col1}$')
                axs[i].set_ylabel(f'${col2}$')
                axs[i].text(0.02, 0.98, subplot_labels[i], transform=axs[i].transAxes, va='top')

            cbar_ax = fig.add_axes([1, 0.05, 0.025, 0.895]) #type: ignore
            fig.colorbar(
                ScalarMappable(norm=norm, cmap='magma'),
                orientation='vertical',
                label='Épocas',
                cax=cbar_ax)
            plt.tight_layout()
            plt.savefig(f'{self.directory}/plots/information_plots.pdf')
            plt.close()

        elif len(combinacoes_sequenciais) == 10:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            axs = axs.ravel()  # Transforma em array 1D para facilitar iteração
            subplot_labels = ['a)', 'b)', 'c)', 'd)']
            indice_vec = [5, 7,8,9]
            for i, j in enumerate(indice_vec):
                col1 = combinacoes_sequenciais[j][0]
                col2 = combinacoes_sequenciais[j][1]
                axs[i].scatter(data[col1], data[col2],c= np.arange(0, size),cmap= 'magma',norm=norm)
                col1 = col1.strip('train/')
                col2 = col2.strip('train/').replace('hat', '\\hat')
                title = f'Relação ${col1}\\times {col2}$'
                axs[i].set_title(title)
                axs[i].set_xlabel(f'${col1}$')
                axs[i].set_ylabel(f'${col2}$')
                axs[i].text(0.02, 0.98, subplot_labels[i], transform=axs[i].transAxes, va='top')

            cbar_ax = fig.add_axes([1, 0.05, 0.025, 0.895]) #type: ignore
            fig.colorbar(
                ScalarMappable(norm=norm, cmap='magma'),
                orientation='vertical',
                label='Épocas',
                cax=cbar_ax)
            plt.tight_layout()
            plt.savefig(f'{self.directory}/plots/information_plots.pdf')
            plt.close()
            
        else:
            for col1, col2 in combinacoes_sequenciais:
                val1 = data[col1] #data.groupby(data.index // self.n_steps)[col1].mean()
                val2 = data[col2] #data.groupby(data.index // self.n_steps)[col2].mean()
                plt.scatter(val1, val2, c= np.arange(0, len(data[col1])), cmap= 'magma')
                col1 = col1.strip('train/')
                col2 = col2.strip('train/')
                title = f'Relação ${col1}\\times {col2}$'
                plt.xlabel(f'${col1}$')
                plt.ylabel(f'${col2}$')
                plt.title(title)
                plt.colorbar(label='Épocas')
                plt.tight_layout()
                plt.savefig(f'{self.directory}/plots/{title}.pdf')
                plt.close()
        
    def treinamento(self):
        seeds = self.fib_seeds
        for seed in tqdm(seeds, desc="Training with different seeds"):
            self.model.set_random_seed(seed)
            self.model.learn(total_timesteps= self.timesteps, progress_bar= True ,callback= CustomCallback(verbose=0, coleta = self.coleta, env_id= self.env_id, directory= self.directory))
        
        ## recompensa
        df = pd.DataFrame(self.model.rewards_list)
        rewards_directory = os.path.join(self.directory, 'rewards.csv')
        df.to_csv(rewards_directory, mode= 'w', index=False, header= True)    

        if not self.recording:
            self.model.save(os.path.join(self.directory, 'agente_treinado'))

        params = {chave: valor for chave, valor in self.__dict__.items() if not chave in ['train_env', 'model', '_hyperparams']}
        json_string = json.dumps(params)
        with open(os.path.join(self.directory, f'{self.env_id}-{self.timesteps}.json'), 'w') as arquivo:      
            arquivo.write(json_string)

        self.plots()
        self.train_env.close()

    def visualizar_modelo(self):
        assert self.model is not None, 'Treine o modelo'
        env = gymnasium.make(self.env_id, render_mode = 'human')
        obs = env.reset()[0]
        i = 0
        seeds = fib(10)[-5:]
        while i < 5:
            self.model.set_random_seed(seeds[i])
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
                i = i + 1
            if keyboard.is_pressed('esc'):
                break
        env.close()

    def coleta_dado(self, n: int):
        assert self.model is not None
        env = gymnasium.make('CartPole-v1')
        dir = os.path.join(self.directory, 'dados.csv')
        self.model.set_random_seed(0)   #* Reprodutibilidade 
        seeds = fib(n)        #* Reprodutibilidade 

        if os.path.exists(dir):
            os.remove(dir)

        for ite, seed in enumerate(seeds):
            obs = env.reset(seed=seed)[0]   #* Reprodutibilidade 
            done = False
            i = 1
            while not done:
                tensor_obs = self.model.policy.obs_to_tensor(obs)[0]
                action1 = self.model.policy.predict_values(tensor_obs)
                action2 = self.model.policy.get_distribution(tensor_obs)
                
                action, _ = self.model.predict(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                df = pd.DataFrame({
                        'Iteração': ite,
                        'Passo': i,
                        'Posição': obs[0],
                        'Velocidade': obs[1],
                        'Ângulo': obs[2],
                        'Velocidade Angular': obs[3],
                        'Distribuição': action2.distribution.probs[0,0].detach().clone().cpu().numpy(), #type: ignore
                        'Entropia': action2.distribution.entropy().detach().clone().cpu().numpy() #type: ignore
                }, index=[0])
                df.to_csv(dir, mode='a' if os.path.exists(dir) else 'w', index=False, header= not os.path.exists(dir))
                done = terminated or truncated
                i += 1
            env.close()