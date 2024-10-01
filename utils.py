from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers.record_video import RecordVideo
from stable_baselines3.common.monitor import Monitor
from npeet import entropy_estimators as ee
from stable_baselines3.ppo import PPO
from torch.nn import functional as F
from pushbullet import Pushbullet
from gymnasium import spaces
import pandas as pd
import numpy as np
import gymnasium
import keyboard
import torch
import os

def alerta():
    with open('chave.txt', 'r') as arquivo:
        chave = arquivo.read()
        pb = Pushbullet(chave)
        pb.push_note("Fim da execução!", "Seu código no computador do laboratório terminou de rodar.")
    print('Done.')

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

class CustomCallback(BaseCallback):
    def __init__(self, coleta: bool, verbose: int = 0):
        super().__init__(verbose)
        self.counter = 0
        self.coleta = coleta

    def _on_step(self) -> bool:
        return super()._on_step()
    
    def _on_rollout_end(self) -> None:
        if not self.coleta:
            return super()._on_rollout_end()
        env = gymnasium.make('CartPole-v1')
        self.model.set_random_seed(0)   #* Reprodutibilidade 
        seeds = fib(100)                #* Reprodutibilidade 

        dir = f'Coleta/coleta_treino_{self.counter}.csv'
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
                mode = 'a' if os.path.exists(dir) else 'w'
                df.to_csv(dir, mode=mode, index=False, header= not os.path.exists(dir))
                done = terminated or truncated
                i += 1
            env.close()
        self.counter += 1

class PPO_tunado(PPO):
        def __init__(self, direc: str, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.direc = direc

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
                mutual_info = [[] for _ in range(5)]
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

                    with torch.no_grad():
                        entrada = self.policy.extract_features(rollout_data.observations)
                        assert (type(entrada) is torch.Tensor)
                        layer1_activations = self.policy.mlp_extractor.policy_net[0](entrada)
                        layer2_activations = self.policy.mlp_extractor.policy_net[2](torch.tanh(layer1_activations))
                        output = self.policy.action_net(torch.tanh(layer2_activations))
                        mutual_info[0].append(ee.mi(tensor_to_numpy(entrada), tensor_to_numpy(layer1_activations)))
                        mutual_info[1].append(ee.mi(tensor_to_numpy(entrada), tensor_to_numpy(layer2_activations)))
                        mutual_info[3].append(ee.mi(tensor_to_numpy(layer1_activations), tensor_to_numpy(layer2_activations)))
                        mutual_info[2].append(ee.mi(tensor_to_numpy(layer1_activations), tensor_to_numpy(output)))
                        mutual_info[4].append(ee.mi(tensor_to_numpy(layer2_activations), tensor_to_numpy(output)))

                # Logs
                for i, medida in enumerate(mutual_info):
                    self.logger.record(f"train/mutual_info_{i}", np.mean(medida))
                self.logger.record("train/entropy_loss", np.mean(entropy_losses))
                self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
                self.logger.record("train/value_loss", np.mean(value_losses))
                self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
                self.logger.record("train/clip_fraction", np.mean(clip_fractions))
                self.logger.record("train/loss", loss.item())
                for i in range(len(self.policy.optimizer.param_groups[0]['params'])):
                    self.logger.record(f"train/gradient_layer_{i}", self.policy.optimizer.param_groups[0]['params'][i].clone().detach().norm().cpu().numpy())
                if hasattr(self.policy, "log_std"):
                    self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
                self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
                self.logger.record("train/clip_range", clip_range)
                if self.clip_range_vf is not None:
                    self.logger.record("train/clip_range_vf", clip_range_vf)

                # --- Início da seção modificada ---
                # CSV writing
                data = self.logger.name_to_value
                df = pd.DataFrame(data, index=[0])

                mode = 'a' if os.path.exists(self.direc) else 'w'
                df.to_csv(self.direc, mode=mode, index=False, header=not os.path.exists(self.direc))
                # --- Fim da seção modificada ---
                
                self._n_updates += 1
                if not continue_training:
                    break

            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
            self.logger.record("train/explained_variance", explained_var)

class Experimento():
    
    def __init__(self, params) -> None:
        self.env_id = 'CartPole-v1'
        self.n_envs = 4
        self.size = [64, 64]
        self.fib_seeds = 5
        self.timesteps = int(1e3) #todo: atualizar isso depois
        self.model = None
        self.recording = False
        self.recording_ep_freq = 100
        self.device = 'auto'
        self.direc = 'resultados.csv'
        self.coleta = False

        for chave, valor in params.items():
            setattr(self, chave, valor)

    def ultimo_indice(self): 
        # Extrair o último índice
        arquivos = os.listdir(self.env_id)
        try:
            nome = arquivos[-1].split('-')
            indice = int(nome[nome.index('serie') + 1]) + 1
            prefix = str(self.env_id) + '-serie-' + str(indice)    
        except:
            prefix = str(self.env_id) + '-serie-' + str(1)

        return prefix
    
    def treinamento(self):
        #* Reprodutibilidade
        torch.manual_seed(0)

        if os.path.exists(self.direc):
            os.remove(self.direc)
        
        # Ambiente de treinamento
        self.train_env = make_vec_env(self.env_id, n_envs= self.n_envs, seed=0, wrapper_class=Monitor)
    
        if self.recording:
            prefix = self.ultimo_indice()
            self.train_env = RecordVideo(self.train_env.get_attr('env')[0], video_folder= self.env_id, name_prefix= prefix, episode_trigger= lambda x: x % self.recording_ep_freq == 0, disable_logger = True)
        self.model = PPO_tunado(self.direc, 'MlpPolicy', self.train_env, policy_kwargs= dict(net_arch = dict(pi=self.size, vf=self.size)), device= self.device) 
        #* Não tenho interesse em deixar o ator e crítico com tamanhos diferentes

        seeds = fib(self.fib_seeds)
        for seed in seeds:
            self.model.set_random_seed(seed)
            self.model.learn(total_timesteps= self.timesteps, callback= CustomCallback(verbose=0, coleta = self.coleta))

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

    def coleta_dado(self, n: int, dir: str = 'coleta.csv'):
        assert self.model is not None
        env = gymnasium.make('CartPole-v1')

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
                mode = 'a' if os.path.exists(dir) else 'w'
                df.to_csv(dir, mode=mode, index=False, header= not os.path.exists(dir))
                done = terminated or truncated
                i += 1
            env.close()