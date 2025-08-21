from stable_baselines3.common.utils import explained_variance
from npeet import entropy_estimators as ee
from torch.nn import functional as F
from stable_baselines3 import PPO
from gymnasium import spaces
from typing import Optional
import pandas as pd
import numpy as np
import gymnasium
import itertools
import torch
import copy
import os

from class_LQR_controller import Controller
from utils import tensor_to_numpy

class PPO_tunado(PPO):
    def __init__(
            self, 
            directory: str, 
            policy: str, 
            env: gymnasium.Env, 
            ref_agent: Optional[str], 
            ref_control: Optional[Controller], 
            calc_mutual_info: bool, 
            hparams: dict
            ):            
        super().__init__(policy, env, **hparams)
        self.directory = os.path.join(directory, 'resultados.csv')
        self.calc_mutual_info = calc_mutual_info
        self.reference_agent = None
        self.reference_control = None
        self.rewards_list = []
        if ref_agent is not None:
            temp_agent = PPO('MlpPolicy', env)
            self.reference_agent = temp_agent.load(ref_agent)
        if ref_control is not None:
            self.reference_control = ref_control
        
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
        
        #%% logging additions
        layer_size = len(self.policy_kwargs['net_arch'])
        measure_size = 2
        vec1 = [f'h_{i+1}' for i in range(layer_size)]
        vec1.insert(0, 'X') 
        vec2 = copy.copy(vec1)
        vec2.extend(['hat Y'])
        
        if self.reference_agent is not None:
            measure_size = measure_size + 2
            vec2.extend(['Y']) # type: ignore
        if self.reference_control is not None:
            measure_size = measure_size + 2
            vec2.extend(['Y_c']) # type: ignore
        
        mutual_info = {}
        ite = 0
        for i, j in enumerate(vec1):
            for k in vec2[i+1:]:
                key = f"I({j},{k})"
                mutual_info[key] = []  
                ite += 1
        # Facilita ao deixar ambos iteráveis
        actor_net = list(
            itertools.chain(
                self.policy.mlp_extractor.policy_net.named_parameters(), 
                self.policy.action_net.named_parameters()
                )
            )
        critic_net = list(
            itertools.chain(
                self.policy.mlp_extractor.value_net.named_parameters(),
                self.policy.value_net.named_parameters()
                )
            )
        
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            metrics = {
                'mutual_info': mutual_info,
                'actor':{
                    'gradient': {key: [0] for key, _ in actor_net},
                    'weights': {key: [value.detach().clone().norm().cpu().numpy()] for key, value in actor_net},
                    'grad_mean': {key: [0] for key, _ in actor_net},
                    'grad_std': {key: [0] for key, _ in actor_net}
                },
                'critic':{
                    'gradient': {key: [0] for key, _ in critic_net},
                    'weights': {key: [value.detach().clone().norm().cpu().numpy()] for key, value in critic_net},
                    'grad_mean': {key: [0] for key, _ in critic_net},
                    'grad_std': {key: [0] for key, _ in critic_net}
                }
            }

            tensor_policy_activations = {key: torch.tensor(0) for key in vec1}
            numpy_policy_activations = {}
            tensor_layers = {key: torch.tensor(0) for key in vec2}
            numpy_layers = {}
            approx_kl_divs = []
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

                if self.calc_mutual_info:
                    with torch.no_grad():
                        entrada = self.policy.extract_features(rollout_data.observations)
                        assert (type(entrada) is torch.Tensor)
                        
                        ite = 1
                        tensor_policy_activations['X'] = entrada
                        x = entrada
                        for i, layer in enumerate(self.policy.mlp_extractor.policy_net):
                            x = layer(x)
                            if i % 2 == 1:  # A cada par de camadas (linear + ativação)
                                tensor_policy_activations[vec1[ite]] = x
                                ite = ite + 1

                        # todo: voltar pra adicionar isso aqui
                        # action_activations = [entrada]
                        # action_layers = [i for i in self.policy.mlp_extractor.value_net]
                        # for i in np.arange(0,len(action_layers),2):
                        #     action_activations.append(action_layers[i+1](action_layers[i](action_activations[int(i/2)])))
                        
                        for key in tensor_policy_activations.keys():
                            tensor_layers[key] = tensor_policy_activations[key]

                        # saídas para comparação
                        #saída da rede
                        tensor_layers['hat Y'] = self.policy.action_net(tensor_policy_activations[key])
                        #saída de controle
                        if self.reference_control is not None:
                            tensor_layers['Y_c'] = torch.tensor(self.reference_control.apply_state_controller(rollout_data.observations))
                        #saída do agente de referência
                        if self.reference_agent is not None:
                            tensor_layers['Y'] = self.reference_agent.predict(rollout_data.observations)[0] #type: ignore

                        for key in tensor_policy_activations.keys():
                            numpy_policy_activations[key] = tensor_to_numpy(tensor_policy_activations[key])
                        for key in tensor_layers.keys():
                            numpy_layers[key] = tensor_to_numpy(tensor_layers[key])
                        
                        
                        for key in metrics['mutual_info']:
                            key1, key2 = key.strip('I()').split(',')
                            metrics['mutual_info'][key].append(ee.mi(numpy_policy_activations[key1], numpy_layers[key2]))

                        # todo: verifica se alterar a ordem clone, grad e detach afeta o resulta de MI
                        for key, value in actor_net:
                            metrics['actor']['weights'][key].append(value.detach().clone().norm().cpu().numpy()) 
                            metrics['actor']['gradient'][key].append(value.grad.detach().clone().norm().cpu().numpy()) #type: ignore
                            metrics['actor']['grad_mean'][key].append(value.grad.detach().clone().mean().norm().cpu().numpy()) #type: ignore
                            if key != 'bias':
                                metrics['actor']['grad_std'][key].append(value.grad.detach().clone().std().cpu().numpy()) #type: ignore
                        
                        for key, value in critic_net:
                            metrics['critic']['weights'][key].append(value.detach().clone().norm().cpu().numpy()) 
                            metrics['critic']['gradient'][key].append(value.grad.detach().clone().norm().cpu().numpy()) #type: ignore
                            metrics['critic']['grad_mean'][key].append(value.grad.detach().clone().mean().norm().cpu().numpy()) #type: ignore
                            if key != 'bias':
                                metrics['critic']['grad_std'][key].append(value.grad.detach().clone().std().cpu().numpy()) #type: ignore
                        

            # Logs
            if self.calc_mutual_info:
                with torch.no_grad():
                    
                    for key, values in metrics['mutual_info'].items():
                        self.logger.record(f"{key}", np.mean(values))

                    self.logger.record("entropy_loss", np.mean(entropy_losses))
                    self.logger.record("policy_gradient_loss", np.mean(pg_losses))
                    self.logger.record("value_loss", np.mean(value_losses))
                    self.logger.record("approx_kl", np.mean(approx_kl_divs))
                    self.logger.record("clip_fraction", np.mean(clip_fractions))
                    self.logger.record("loss", loss.item())
                    for key in metrics['actor']['gradient'].keys():
                        self.logger.record(f"actor_weight_layer_{key}", np.mean(metrics['actor']['weights'][key]))
                        self.logger.record(f"actor_grad_layer_{key}", np.mean(metrics['actor']['gradient'][key]))
                        self.logger.record(f"actor_mean_grad_layer_{key}", np.mean(metrics['actor']['grad_mean'][key]))
                        self.logger.record(f"actor_std_grad_layer_{key}", np.mean(metrics['actor']['grad_std'][key]))
                    
                    for key in metrics['critic']['gradient'].keys():
                        self.logger.record(f"critic_weight_layer_{key}", np.mean(metrics['critic']['weights'][key]))
                        self.logger.record(f"critic_grad_layer_{key}", np.mean(metrics['critic']['gradient'][key]))
                        self.logger.record(f"critic_mean_grad_layer_{key}", np.mean(metrics['critic']['grad_mean'][key]))
                        self.logger.record(f"critic_std_grad_layer_{key}", np.mean(metrics['critic']['grad_std'][key]))

                    if hasattr(self.policy, "log_std"):
                        self.logger.record("std", torch.exp(self.policy.log_std).mean().item())

                    if self.clip_range_vf is not None:
                        self.logger.record("clip_range_vf", clip_range_vf)
                
                # CSV writing
                data = self.logger.name_to_value
                df = pd.DataFrame(data, index=[0])

                df.to_csv(self.directory, mode='a' if os.path.exists(self.directory) else 'w', index=False, header= not os.path.exists(self.directory))
            
            self._n_updates += 1
            if not continue_training:
                break
        
        reward = self.env.envs[0].get_episode_rewards() #type: ignore
        self.rewards_list.append(reward) 
        self.env.envs[0].episode_returns = [] #type: ignore
        
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("explained_variance", explained_var)