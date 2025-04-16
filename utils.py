from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers.record_video import RecordVideo
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv
from npeet import entropy_estimators as ee
from stable_baselines3.ppo import PPO
from torch.nn import functional as F
from typing import Union, Optional
from pushbullet import Pushbullet
import matplotlib.pyplot as plt
from gymnasium import spaces
from scipy import linalg
import pandas as pd
import numpy as np
import gymnasium
import itertools
import keyboard
import torch
import copy 
import json
import os

plt.style.use('style.mplstyle')

# usar com o decorador @wrap_alerta
def wrap_alerta(func):
    """função para alertar via pushbullet
    """
    def msg(key: int, f):
        with open('chave.txt', 'r') as arquivo:
            chave = arquivo.read()
            pb = Pushbullet(chave)
            title = 'Fim da execução!' if key == 1 else 'Deu ruim!'
            pb.push_note(title, f)        

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg(0, e)
        finally:
            msg(1,'cabou')
    return wrapper

def alerta():
    with open('chave.txt', 'r') as arquivo:
        chave = arquivo.read()
        pb = Pushbullet(chave)
        title = 'Fim da execução!'
        pb.push_note(title, 'cabou') 


def fib(n: int):
    """Gera os n primeiros números da sequência de Fibonacci utilizando um loop.

    Args:
        n: Número de termos da sequência.

    Returns:
        Uma lista com os n primeiros números da sequência.
    """
    assert n > 0, 'N precisa ser maior do que 0'
    if n == 1:
        return [1]
    fib = [1, 2]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

def tensor_to_numpy(val:torch.Tensor):
    return val.clone().detach().cpu().numpy()

class Controller:
    '''Implements a feedback controller'''
    def __init__(self, environment):
        # gravity
        self.g = 10
        # pole length
        self.lp = environment.env.get_wrapper_attr('length')
        # pole mass
        self.mp = environment.env.get_wrapper_attr('masspole')
        # cart mass
        self.mk = environment.env.get_wrapper_attr('masscart')
        # total mass
        self.mt = environment.env.get_wrapper_attr('total_mass')
        
    def state_controller(self):
        # state matrix
        a = self.g/(self.lp*(4.0/3 - self.mp/(self.mp+self.mk)))
        A = np.array([[0, 1, 0, 0],
            [0, 0, a, 0],
            [0, 0, 0, 1],
            [0, 0, a, 0]])
            
        # input matrix
        b = -1/(self.lp*(4.0/3 - self.mp/(self.mp+self.mk)))
        B = np.array([[0], [1/self.mt], [0], [b]])
        
        # choose R (weight for input)
        R = np.eye(1, dtype=int)
        # choose Q (weight for state)
        Q = 5*np.eye(4, dtype=int)
        
        # solve ricatti equation
        P = linalg.solve_continuous_are(A, B, Q, R)     #! Tem algo aqui que crasha o kernel
        
        # calculate optimal controller gain
        K = np.dot(np.linalg.inv(R), np.dot(B.T, P))

        return K
        
    def apply_state_controller(self, x):
        # Adaptei pro modo torch
        K = self.state_controller()
        
        # Calcular a força para todos os estados em um único passo
        forces = -torch.matmul(x, torch.tensor(K.astype(np.float32).T))
        
        # Determinar a ação (0 ou 1) com base na força
        actions = torch.where(forces > 0, torch.ones_like(forces), torch.zeros_like(forces))

        return actions


class CustomCallback(BaseCallback):
    def __init__(self, coleta: bool, env_id: str, direc: str, verbose: int = 0):
        super().__init__(verbose)
        self.counter = 0
        self.coleta = coleta
        self.env_id = env_id
        self.direc = direc

    def _on_step(self) -> bool:
        return super()._on_step()
    
    def _on_rollout_start(self) -> None:
        if not self.coleta:
            return super()._on_rollout_end()
        env = gymnasium.make(self.env_id)
        self.model.set_random_seed(0)   #* Reprodutibilidade 
        seeds = fib(100)                #* Reprodutibilidade 
        os.makedirs(os.path.join(self.direc, 'Coleta'))
        dir = os.path.join(self.direc,f'Coleta/coleta_treino_{str(self.counter).zfill(3)}.csv')
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
                        'Distribuição': action2.distribution.probs[0,0].clone().detach().cpu().numpy(), #type: ignore
                        'Entropia': action2.distribution.entropy().clone().detach().cpu().numpy() #type: ignore
                                }, index=[0])
                df.to_csv(dir, mode= 'a' if os.path.exists(dir) else 'w', index=False, header= not os.path.exists(dir))
                done = terminated or truncated
                i += 1
            env.close()
        self.counter += 1

class PPO_tunado(PPO):
        def __init__(self, direc: str, policy: str, env: gymnasium.Env, ref_agent: Optional[str], ref_control: Optional[Controller], calc_mi: bool, hparams: dict):            
            super().__init__(policy, env, **hparams)
            self.direc = os.path.join(direc, 'resultados.csv')
            self.calc_mi = calc_mi
            self.reference_agent = None
            self.control = None
            self.rewards_list = []
            if ref_agent is not None:
                temp_agent = PPO('MlpPolicy', env)
                self.reference_agent = temp_agent.load(ref_agent)
            if ref_control is not None:
                self.control = ref_control
            
        def train(self):
            """
            Update policy using the currently gathered rollout buffer.
            """
            # Switch to train mode (this affects batch norm / dropout)
            self.policy.set_training_mode(True)
            # Update optimizer learning rate
            self._update_learning_rate(self.policy.optimizer)
            # Compute current clip range
            clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
            # Optional: clip range for the value function
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

            entropy_losses = []
            pg_losses, value_losses = [], []
            clip_fractions = []

            continue_training = True
            # train for n_epochs epochs
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                layer_size = len(self.policy_kwargs['net_arch'])
                measure_size = 2
                if self.reference_agent is not None:
                    measure_size = measure_size + 2
                if self.control is not None:
                    measure_size = measure_size + 2
                mi_size = int((layer_size + measure_size)*(layer_size + 1)/2)
                

                mutual_info = [[] for _ in range(mi_size)]
                actor_grad_info = [[] for _ in range(2*(layer_size + 1))]
                critic_grad_info = [[] for _ in range(2*(layer_size + 1))]
                actor_weights_info = [[] for _ in range(2*(layer_size + 1))]
                critic_weights_info = [[] for _ in range(2*(layer_size + 1))]

                grad_info = [[] for _ in range(4*(layer_size + 1))]
                weights_info = [[] for _ in range(4*(layer_size + 1))]

                # Do a complete pass on the rollout buffer
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        
                        values_pred = rollout_data.old_values + torch.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf #type: ignore
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -torch.mean(-log_prob)
                    else:
                        entropy_loss = -torch.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with torch.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                    if self.calc_mi:
                        with torch.no_grad():
                            entrada = self.policy.extract_features(rollout_data.observations)
                            assert (type(entrada) is torch.Tensor)
                            
                            # Ok, isso funciona para extrair as ativações intermediárias
                            # favor não mexer
                            x = entrada
                            policy_activations = [x]
                            for i, layer in enumerate(self.policy.mlp_extractor.policy_net):
                                x = layer(x)
                                if i % 2 == 1:  # A cada par de camadas (linear + ativação)
                                    policy_activations.append(x)


                            # action_activations = [entrada]
                            # action_layers = [i for i in self.policy.mlp_extractor.value_net]
                            # for i in np.arange(0,len(action_layers),2):
                            #     action_activations.append(action_layers[i+1](action_layers[i](action_activations[int(i/2)])))
                            
                            # saídas para comparação
                            layers = copy.copy(policy_activations)
                            #saída da rede
                            layers.append(self.policy.action_net(policy_activations[-1]))
                            #saída de controle
                            if self.control is not None:
                                layers.append(self.control.apply_state_controller(rollout_data.observations))
                            #saída do agente de referência
                            if self.reference_agent is not None:
                                layers.append(self.reference_agent.predict(rollout_data.observations)[0]) #type: ignore

                            policy_activations = [tensor_to_numpy(i) for i in policy_activations]
                            layers = [tensor_to_numpy(i) for i in layers]
                            
                            ite = 0
                            for i, j in enumerate(policy_activations):
                                for k in layers[i+1:]:
                                    mutual_info[ite].append(ee.mi(j, k))
                                    ite = ite + 1
                                                            
                            for i in range(layer_size + 1):
                                grad_info[i].append(self.policy.optimizer.param_groups[0]['params'][i].grad.clone().detach().norm().cpu().numpy())
                                weights_info[i].append(self.policy.optimizer.param_groups[0]['params'][i].clone().detach().norm().cpu().numpy())

                            # coletando os pesos e gradientes
                            ## rede actor
                            for i, j in enumerate((itertools.chain(self.policy.mlp_extractor.policy_net.named_parameters(), self.policy.action_net.named_parameters()))):
                                actor_weights_info[i].append(j[1].clone().detach().norm().cpu().numpy()) 
                                actor_grad_info[i].append(j[1].grad.clone().detach().norm().cpu().numpy()) #type: ignore
                            
                            ## rede critic
                            for i, j in enumerate(itertools.chain(self.policy.mlp_extractor.value_net.named_parameters(), self.policy.value_net.named_parameters())):
                                critic_weights_info[i].append(j[1].clone().detach().norm().cpu().numpy()) 
                                critic_grad_info[i].append(j[1].grad.clone().detach().norm().cpu().numpy()) #type: ignore
                # Logs
                if self.calc_mi:
                    with torch.no_grad():
                        vec1 = [f'h_{i+1}' for i in range(layer_size)]
                        vec1.insert(0, 'X')
                        vec2 = copy.copy(vec1)
                        vec2.extend(['hat Y'])

                        if self.control is not None:
                            vec2.extend(['Y_c']) # type: ignore
                        if self.reference_agent is not None:
                            vec2.extend(['Y']) # type: ignore
                        
                        # todo checar pra ver se tá certo isso
                        ite = 0
                        for i, j in enumerate(vec1):
                            for k in vec2[i+1:]:
                                self.logger.record(f"train/I({j},{k})", np.mean(mutual_info[ite]))
                                ite = ite + 1

                        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
                        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
                        self.logger.record("train/value_loss", np.mean(value_losses))
                        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
                        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
                        self.logger.record("train/loss", loss.item())
                        for i in range(2*(layer_size + 1)):
                            self.logger.record(f"train/actor_weight_layer_{i}", np.mean(actor_weights_info[i]))
                            self.logger.record(f"train/actor_grad_layer_{i}", np.mean(actor_grad_info[i]))
                        
                        for i in range(2*(layer_size + 1)):
                            self.logger.record(f"train/critic_weight_layer_{i}", np.mean(critic_weights_info[i]))
                            self.logger.record(f"train/critic_grad_layer_{i}", np.mean(critic_grad_info[i]))

                        for i in range(layer_size + 1):
                            self.logger.record(f"train/gradient_layer_{i}", np.mean(grad_info[i]))
                            self.logger.record(f"train/weights_layer_{i}", np.mean(weights_info[i]))
                        if hasattr(self.policy, "log_std"):
                            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

                        if self.clip_range_vf is not None:
                            self.logger.record("train/clip_range_vf", clip_range_vf)
                    
                    # CSV writing
                    data = self.logger.name_to_value
                    df = pd.DataFrame(data, index=[0])

                    df.to_csv(self.direc, mode='a' if os.path.exists(self.direc) else 'w', index=False, header= not os.path.exists(self.direc))
                
                self._n_updates += 1
                if not continue_training:
                    break
            
            #! ou vai quebrar o treinamento ou vai ser uma jogada de mestre
            self.rewards_list.append(self.env.envs[0].get_episode_rewards()) #type: ignore
            self.env.envs[0].episode_returns = [] #type: ignore

            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
            self.logger.record("train/explained_variance", explained_var)

class Experimento():
    
    def __init__(self, params) -> None:
        # Setagem
        self.env_id = 'CartPole-v1'
        self.n_envs = 4
        self.fib_seeds = fib(5)
        self.timesteps = int(1e3) 
        self.reference_agent = None
        self.controller = None
        self.calc_mi = False
        self.control = False

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
        self.device = "auto"
        
        # Recording Parameters
        self.recording = False
        self.recording_ep_freq = 100
        self.device = 'auto'
        self.coleta = False
        # Fim da setagem

        for chave, valor in params.items():
            setattr(self, chave, valor)
        
        #isso serve pra criar uma pasta com número maior sem perder a ordenação
        if not os.path.exists(f'{self.direc}/000'):
            os.makedirs(f'{self.direc}/000')
            self.direc = f'{self.direc}/000'
        else:
            last_number = os.listdir(self.direc)[-1]

            self.direc = os.path.join(self.direc, str(int(last_number) + 1).zfill(3))
            os.makedirs(self.direc)

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

        if self.control:
            self.controller = Controller(self.env.get_attr('env')[0]) #type: ignore

        #* Reprodutibilidade
        torch.manual_seed(0)
        
        self.model = PPO_tunado(self.direc, 'MlpPolicy', self.train_env, self.reference_agent, self.controller, self.calc_mi, self._hyperparams) 
    
    def plots(self):
        os.makedirs(f'{self.direc}/plots')
        data = pd.read_csv(f'{self.direc}/resultados.csv')
        col_info = {col: [item.strip() for item in col.strip('train/I()').split(',')] for col in data.columns}
        combinacoes_sequenciais = []

        for col1, info1 in col_info.items():
            if len(info1) == 2:  # Certifica-se de que a coluna tem dois elementos (para ter um "segundo")
                segundo_elemento_col1 = info1[1]
                for col2, info2 in col_info.items():
                    if col1 != col2 and len(info2) == 2:  # Evita comparar a mesma coluna e garante dois elementos
                        primeiro_elemento_col2 = info2[0]
                        if segundo_elemento_col1 == primeiro_elemento_col2:
                            combinacoes_sequenciais.append((col1, col2))

        if combinacoes_sequenciais:
            print("Gerando plots para combinações sequenciais encontradas:")
            for col1, col2 in combinacoes_sequenciais:
                plt.scatter(data[col1], data[col2], c= np.arange(0, len(data[col1])), cmap= 'magma')
                col1 = col1.strip('train/')
                col2 = col2.strip('train/')
                title = f'Relação ${col1}-{col2}$'
                plt.xlabel(f'${col1}$')
                plt.ylabel(f'${col2}$')
                plt.title(title)
                plt.colorbar(label='Epochs')
                plt.tight_layout()
                plt.savefig(f'{self.direc}/plots/{title.strip("$/$")}.pdf')
                plt.show()
        else:
            print("Nenhuma combinação sequencial encontrada para plotar.")    

    def treinamento(self):
        # Ambiente de treinamento

        seeds = self.fib_seeds
        for seed in seeds:
            self.model.set_random_seed(seed)
            self.model.learn(total_timesteps= self.timesteps, callback= CustomCallback(verbose=0, coleta = self.coleta, env_id= self.env_id, direc= self.direc))

        if not self.recording:
            self.model.save(os.path.join(self.direc, 'agente_treinado'))

        params = {chave: valor for chave, valor in self.__dict__.items() if not chave in ['train_env', 'model']}
        json_string = json.dumps(params)
        with open(os.path.join(self.direc, 'hparams.json'), 'w') as arquivo:      
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
        dir = os.path.join(self.direc, 'dados.csv')
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
                        'Distribuição': action2.distribution.probs[0,0].clone().detach().cpu().numpy(), #type: ignore
                        'Entropia': action2.distribution.entropy().clone().detach().cpu().numpy() #type: ignore
                }, index=[0])
                df.to_csv(dir, mode='a' if os.path.exists(dir) else 'w', index=False, header= not os.path.exists(dir))
                done = terminated or truncated
                i += 1
            env.close()