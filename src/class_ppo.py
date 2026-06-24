from typing import Optional

import gymnasium
import torch
from gymnasium import spaces
from src.class_fetcher import ActivationFetcher
from src.class_mutual_info import MutualInfoCalculator
from src.class_parameter_tracker import ParameterTracker
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


class CustomPPO(PPO):
    def __init__(
        self,
        directory: str,
        policy: str,
        env: gymnasium.Env,
        ref_agent: Optional[str],
        calc_mutual_info: bool,
        hparams: dict,
    ):
        super().__init__(policy, env, **hparams)
        self.rewards_list = []
        self.reference_agent = None

        if ref_agent is not None:
            temp_agent = PPO("MlpPolicy", env)
            self.reference_agent = temp_agent.load(ref_agent)

        self.fetcher = ActivationFetcher(self) if calc_mutual_info else None

        if calc_mutual_info:
            layer_size = len(self.policy_kwargs["net_arch"])
            has_ref = self.reference_agent is not None
            self.mi_calculator = MutualInfoCalculator(self, layer_size, has_ref)
            self.tracker = ParameterTracker(self, directory)
        else:
            self.mi_calculator = None
            self.tracker = None

    def train(self):
        self.policy.set_training_mode(True)
        if self.fetcher is not None:
            self.fetcher.attach_hooks()

        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)

        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses = [], [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            mi_keys = self.mi_calculator.mapping.keys() if self.mi_calculator else {}
            metrics = self.tracker.create_empty_metrics(mi_keys) if self.tracker else {}
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
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
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -torch.mean(-log_prob) if entropy is None else -torch.mean(entropy)
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

                if self.fetcher is not None and self.mi_calculator is not None:
                    self.mi_calculator.compute(metrics, self.fetcher, rollout_data.observations)
                    self.tracker.capture_norms(metrics)
                    self.fetcher.clear()

            # Consolidação das perdas enviada diretamente ao tracker
            if self.tracker:
                self.tracker.collect_epoch_metrics(
                    metrics,
                    epoch_losses={
                        "entropy_loss": entropy_losses,
                        "policy_gradient_loss": pg_losses,
                        "value_loss": value_losses,
                        "approx_kl": approx_kl_divs,
                        "clip_fraction": clip_fractions,
                        "loss": [loss.item()],
                        "clip_range_vf_val": clip_range_vf,
                    },
                )

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        self.logger.record("explained_variance", explained_var)

        if self.tracker:
            self.tracker.collect_reward_metrics()
            self.tracker.flush_logs_to_disk()

        if self.fetcher is not None:
            self.fetcher.remove()
