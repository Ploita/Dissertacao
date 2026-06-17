import itertools
import os
from typing import Optional

import gymnasium
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from class_fetcher import ActivationFetcher
from class_mutual_info import MutualInfoCalculator


class CustomPPO(PPO):
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

        if ref_agent is not None:
            temp_agent = PPO('MlpPolicy', env)
            self.reference_agent = temp_agent.load(ref_agent)

        self.fetcher = ActivationFetcher(self) if calc_mutual_info else None

        if calc_mutual_info:
            layer_size = len(self.policy_kwargs['net_arch'])
            has_ref = self.reference_agent is not None
            self.mi_calculator = MutualInfoCalculator(self, layer_size, has_ref)
        else:
            self.mi_calculator = None


    def train(self):
        self.policy.set_training_mode(True)
        if self.fetcher is not None:
            self.fetcher.attach_hooks()

        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True

        actor_net = list(itertools.chain(
            self.policy.mlp_extractor.policy_net.named_parameters(),
            self.policy.action_net.named_parameters())
        )
        critic_net = list(itertools.chain(
            self.policy.mlp_extractor.value_net.named_parameters(),
            self.policy.value_net.named_parameters())
        )

        for epoch in range(self.n_epochs):
            mi_keys = self.mi_calculator.mapping.keys() if self.mi_calculator else {}
            metrics = {
                "actor": {
                    "mutual_info": {k: [] for k in mi_keys},
                    "gradient": {k: [] for k, _ in actor_net},
                    "weights": {k: [v.norm().item()] for k, v in actor_net},
                    "grad_mean": {k: [] for k, _ in actor_net},
                    "grad_std": {k: [] for k, _ in actor_net},
                },
                "critic": {
                    "mutual_info": {k: [] for k in mi_keys},
                    "gradient": {k: [] for k, _ in critic_net},
                    "weights": {k: [v.norm().item()] for k, v in critic_net},
                    "grad_mean": {k: [] for k, _ in critic_net},
                    "grad_std": {k: [] for k, _ in critic_net},
                },
            }

            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # Os hooks disparam aqui nativamente de forma limpa
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions
                    )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # Delegando o cálculo para a classe de MI, isolando os Tensores
                if self.fetcher is not None and self.mi_calculator is not None:
                    self.mi_calculator.compute(metrics, self.fetcher, rollout_data.observations)

                    for key, value in actor_net:
                        metrics['actor']['weights'][key].append(value.norm().item())
                        metrics['actor']['gradient'][key].append(
                            value.grad.norm().item() if value.grad is not None else 0.0
                        )

                    for key, value in critic_net:
                        metrics['critic']['weights'][key].append(value.norm().item())
                        metrics['critic']['gradient'][key].append(
                            value.grad.norm().item() if value.grad is not None else 0.0
                        )

                    self.fetcher.clear()

            if self.fetcher is not None and self.mi_calculator is not None:
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
                        self.logger.record(
                            f"actor_weight_layer_{key}",
                            np.mean(metrics['actor']['weights'][key])
                        )
                        self.logger.record(
                            f"actor_grad_layer_{key}",
                            np.mean(metrics['actor']['gradient'][key])
                        )

                    for key in metrics['critic']['gradient'].keys():
                        self.logger.record(
                            f"critic_weight_layer_{key}",
                            np.mean(metrics['critic']['weights'][key])
                        )
                        self.logger.record(
                            f"critic_grad_layer_{key}",
                            np.mean(metrics['critic']['gradient'][key])
                        )

                    if hasattr(self.policy, "log_std"):
                        self.logger.record(
                            "policy_log_std",
                            torch.exp(self.policy.log_std).mean().item()
                        )

                    if self.clip_range_vf is not None:
                        self.logger.record("clip_range_vf", clip_range_vf)

                data = self.logger.name_to_value
                df = pd.DataFrame(data, index=[0])
                df.to_csv(
                    self.directory,
                    mode='a' if os.path.exists(self.directory) else 'w',
                    index=False,
                    header=not os.path.exists(self.directory)
                )

            self._n_updates += 1
            if not continue_training:
                break

        reward = self.env.envs[0].get_episode_rewards()
        self.rewards_list.append(reward)
        self.env.envs[0].episode_returns = []

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten()
        )
        self.logger.record("explained_variance", explained_var)

        if self.fetcher is not None:
            self.fetcher.remove()
