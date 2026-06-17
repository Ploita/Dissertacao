import copy
import itertools
import os
from typing import Optional

import gymnasium
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from npeet import entropy_estimators as ee
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

class ActivationFetcher:
    def __init__(self, model):
        self.activations = {}
        self.handles = []
        
        # Dicionário mapeando o nome da rede para o objeto nn.Sequential do SB3
        networks = {
            "actor": model.policy.mlp_extractor.policy_net,
            "critic": model.policy.mlp_extractor.value_net
        }
        
        # Correção no Fetcher para espelhar seu loop antigo por índice ímpar:
        for net_name, net in networks.items():
            layer_idx = 1
            for i, module in enumerate(net):
                if i % 2 == 1:  # Captura exatamente no mesmo ponto do seu código original
                    hook_name = f"{net_name}_h_{layer_idx}"
                    self.handles.append(module.register_forward_hook(self.get_hook(hook_name)))
                    layer_idx += 1

    def get_hook(self, name):
        # Esta função é disparada nativamente pelo PyTorch após o forward da camada
        def hook(module, input, output):
            # Clonamos o tensor, extraímos do grafo computacional, mandamos para CPU e convertemos em NumPy
            self.activations[name] = output.detach().cpu().numpy()
        return hook

    def clear(self):
        """Limpa o dicionário para o próximo mini-batch."""
        self.activations = {}

    def remove(self):
        """Remove os hooks da rede para liberar memória caso o treino acabe."""
        for h in self.handles:
            h.remove()

class PPO_tunado(PPO):
    def __init__(
            self,
            directory: str,
            policy: str,
            env: gymnasium.Env,
            ref_agent: Optional[str],
            calc_mutual_info: bool,
            hparams: dict
            ):
        super().__init__(policy, env, **hparams)
        self.directory = os.path.join(directory, 'resultados.csv')
        self.reference_agent = None
        self.reference_control = None
        self.rewards_list = []
        self.fetcher = ActivationFetcher(self) if calc_mutual_info else None
        if ref_agent is not None:
            temp_agent = PPO('MlpPolicy', env)
            self.reference_agent = temp_agent.load(ref_agent)


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
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining
            )  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        layer_size = len(self.policy_kwargs['net_arch'])
        measure_size = 2

        # Nomes RAW (brutos) - Usados para indexar as ativações internamente (como antes)
        layer_names_raw = [f'h_{i+1}' for i in range(layer_size)]
        layer_names_raw.insert(0, 'X')
        output_names_raw = copy.copy(layer_names_raw)
        output_names_raw.extend(['hat Y'])

        # Mapeamento de nomes RAW para SAFE (Seguros para o CSV/Logging)
        def get_safe_name(name: str) -> str:
            """Converte nomes de camada (X, h_1, hat Y, Y) em nomes seguros (X, h1, Yhat, Y_ref)."""
            if name == 'Y':
                return 'Y_ref'
            return name.replace('h_', 'h').replace('hat Y', 'Yhat')

        raw_to_safe_map = {raw: get_safe_name(raw) for raw in output_names_raw}

        if self.reference_agent is not None:
            measure_size = measure_size + 2
            output_names_raw.extend(['Y']) # type: ignore
            # Adiciona o agente de referência ao mapeamento seguro
            raw_to_safe_map['Y'] = 'Y_ref'

        # MI MAPPING: Mapeia a chave segura (para logging) para o par de chaves brutas (para lookup)
        mutual_info_mapping = {}
        for i, j_raw in enumerate(layer_names_raw):
            for k_raw in output_names_raw[i+1:]:
                j_safe = raw_to_safe_map.get(j_raw, j_raw)
                k_safe = raw_to_safe_map.get(k_raw, k_raw)

                key_safe = f"I_{j_safe}_{k_safe}" # Novo formato: I_X_h1
                # (Raw Name 1, Raw Name 2) para lookup
                mutual_info_mapping[key_safe] = (j_raw, k_raw)

        # Inicializa o dicionário de MI usando as chaves SEGURAS
        mutual_info = {key_safe: [] for key_safe in mutual_info_mapping.keys()}

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
                'actor':{
                    'mutual_info': {key: [] for key in mutual_info.keys()}, # Usa chaves SEGURAS
                    'gradient': {key: [] for key, _ in actor_net},
                    'weights': {key: [value.norm().item()] for key, value in actor_net},
                    'grad_mean': {key: [] for key, _ in actor_net},
                    'grad_std': {key: [] for key, _ in actor_net}
                },
                'critic':{
                    'mutual_info': {key: [] for key in mutual_info.keys()}, # Usa chaves SEGURAS
                    'gradient': {key: [] for key, _ in critic_net},
                    'weights': {key: [value.norm().item()] for key, value in critic_net},
                    'grad_mean': {key: [] for key, _ in critic_net},
                    'grad_std': {key: [] for key, _ in critic_net}
                }
            }

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

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions
                    )
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
                        values - rollout_data.old_values,
                        -clip_range_vf, #type: ignore
                        clip_range_vf
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
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch}"
                              f"due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                if self.fetcher is not None:
                    with torch.no_grad():
                        features = self.policy.extract_features(rollout_data.observations)
                        if not isinstance(features, torch.Tensor):
                            raise TypeError(f"Expected torch.Tensor, got {type(features)}")
                        
                        self.fetcher.activations["actor_X"] = features.cpu().numpy()
                        self.fetcher.activations["critic_X"] = features.cpu().numpy()
                        
                        self.fetcher.activations["actor_hat Y"] = self.policy.action_net(self.policy.mlp_extractor.policy_net(features)).cpu().numpy()
                        self.fetcher.activations["critic_hat Y"] = self.policy.value_net(self.policy.mlp_extractor.value_net(features)).cpu().numpy()

                        if self.reference_agent is not None:
                            ref_out = self.reference_agent.predict(rollout_data.observations)[0]
                            # Converte para array float estável compatível com a entrada
                            ref_out_tensor = torch.from_numpy(ref_out).to(self.device)
                            self.fetcher.activations["actor_Y"] = ref_out_tensor.cpu().numpy()
                            self.fetcher.activations["critic_Y"] = ref_out_tensor.cpu().numpy()

                        for key_safe, (raw1, raw2) in mutual_info_mapping.items():
                            #O dicionário 'self.fetcher.activations' agora possui chaves como:
                            # 'actor_X', 'actor_h_1', 'actor_hat Y', 'actor_Y'
                            
                            # MI do Ator
                            metrics['actor']['mutual_info'][key_safe].append(ee.mi(
                                self.fetcher.activations[f"actor_{raw1}"],
                                self.fetcher.activations[f"actor_{raw2}"]
                            ))

                            # MI do Crítico
                            metrics['critic']['mutual_info'][key_safe].append(ee.mi(
                                self.fetcher.activations[f"critic_{raw1}"],
                                self.fetcher.activations[f"critic_{raw2}"]
                            ))

                        for key, value in actor_net:
                            metrics['actor']['weights'][key].append(value.norm().item())
                            if value.grad is not None:
                                metrics['actor']['gradient'][key].append(value.grad.norm().item())
                            else:
                                metrics['actor']['gradient'][key].append(0.0)

                        for key, value in critic_net:
                            metrics['critic']['weights'][key].append(value.norm().item())
                            if value.grad is not None:
                                metrics['critic']['gradient'][key].append(value.grad.norm().item())
                            else:
                                metrics['critic']['gradient'][key].append(0.0)

            # Logs
            if self.fetcher is not None:
                with torch.no_grad():
                    for key_safe, values1 in metrics['actor']['mutual_info'].items():
                        self.logger.record(f"actor_{key_safe}", np.mean(values1))

                    for key_safe, values2 in metrics['critic']['mutual_info'].items():
                        self.logger.record(f"critic_{key_safe}", np.mean(values2))

                    self.logger.record("entropy_loss", np.mean(entropy_losses))
                    self.logger.record("policy_gradient_loss", np.mean(pg_losses))
                    self.logger.record("value_loss", np.mean(value_losses))
                    self.logger.record("approx_kl", np.mean(approx_kl_divs))
                    self.logger.record("clip_fraction", np.mean(clip_fractions))
                    self.logger.record("loss", loss.item())
                    for key in metrics['actor']['gradient'].keys():
                        self.logger.record(f"actor_weight_layer_{key}", np.mean(metrics['actor']['weights'][key]))
                        self.logger.record(f"actor_grad_layer_{key}", np.mean(metrics['actor']['gradient'][key]))

                    for key in metrics['critic']['gradient'].keys():
                        self.logger.record(f"critic_weight_layer_{key}", np.mean(metrics['critic']['weights'][key]))
                        self.logger.record(f"critic_grad_layer_{key}", np.mean(metrics['critic']['gradient'][key]))

                    if hasattr(self.policy, "log_std"):
                        self.logger.record("policy_log_std", torch.exp(self.policy.log_std).mean().item())

                    if self.clip_range_vf is not None:
                        self.logger.record("clip_range_vf", clip_range_vf)

                # CSV writing
                data = self.logger.name_to_value
                df = pd.DataFrame(data, index=[0])

                df.to_csv(self.directory, mode='a' if os.path.exists(self.directory) else 'w', index=False, header= not os.path.exists(self.directory))

            self._n_updates += 1
            if not continue_training:
                break

        #! Quando for paralelizar, investigar como coletar adequadamente essa recompensa
        reward = self.env.envs[0].get_episode_rewards() #type: ignore
        self.rewards_list.append(reward)
        self.env.envs[0].episode_returns = [] #type: ignore

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("explained_variance", explained_var)
