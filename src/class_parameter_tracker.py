import itertools
import os
from typing import Iterable

import numpy as np
import pandas as pd
import torch


class ParameterTracker:
    """Responsabilidade Única: Estruturar, capturar e persistir métricas e logs."""

    def __init__(self, model, directory: str):
        self.model = model
        self.metrics_directory = os.path.join(directory, "resultados.csv")
        self.rewards_directory = os.path.join(directory, "rewards.csv")
        self.eval_logs = []
        self.reward_logs = []

        # Mapeamento interno das redes
        self.actor_net = list(
            itertools.chain(
                model.policy.mlp_extractor.policy_net.named_parameters(),
                model.policy.action_net.named_parameters(),
            )
        )
        self.critic_net = list(
            itertools.chain(
                model.policy.mlp_extractor.value_net.named_parameters(),
                model.policy.value_net.named_parameters(),
            )
        )

    def create_empty_metrics(self, mi_keys: Iterable) -> dict:
        """Gera a estrutura inicial do dicionário de métricas da época."""
        return {
            "actor": {
                "mutual_info": {k: [] for k in mi_keys},
                "gradient": {k: [] for k, _ in self.actor_net},
                "weights": {k: [v.norm().item()] for k, v in self.actor_net},
            },
            "critic": {
                "mutual_info": {k: [] for k in mi_keys},
                "gradient": {k: [] for k, _ in self.critic_net},
                "weights": {k: [v.norm().item()] for k, v in self.critic_net},
            },
        }

    def capture_norms(self, metrics: dict):
        """Captura as normas dos pesos e gradientes atuais do ator e crítico."""
        for key, value in self.actor_net:
            metrics["actor"]["weights"][key].append(value.norm().item())
            metrics["actor"]["gradient"][key].append(
                value.grad.norm().item() if value.grad is not None else 0.0
            )

        for key, value in self.critic_net:
            metrics["critic"]["weights"][key].append(value.norm().item())
            metrics["critic"]["gradient"][key].append(
                value.grad.norm().item() if value.grad is not None else 0.0
            )

    def collect_epoch_metrics(self, metrics: dict, epoch_losses: dict):
        """Processa e registra as métricas estatísticas ao fim de cada época."""
        with torch.no_grad():
            # 1. Informação Mútua
            for key_safe, values in metrics["actor"]["mutual_info"].items():
                self.model.logger.record(f"actor_{key_safe}", np.mean(values))
            for key_safe, values in metrics["critic"]["mutual_info"].items():
                self.model.logger.record(f"critic_{key_safe}", np.mean(values))

            # 2. Perdas (Losses) obtidas no treinamento
            for loss_name, loss_values in epoch_losses.items():
                if loss_values is not None:
                    self.model.logger.record(loss_name, np.mean(loss_values))

            # 3. Pesos e Gradientes do Ator
            for key in metrics["actor"]["gradient"].keys():
                self.model.logger.record(
                    f"actor_weight_layer_{key}", np.mean(metrics["actor"]["weights"][key])
                )
                self.model.logger.record(
                    f"actor_grad_layer_{key}", np.mean(metrics["actor"]["gradient"][key])
                )

            # 4. Pesos e Gradientes do Crítico
            for key in metrics["critic"]["gradient"].keys():
                self.model.logger.record(
                    f"critic_weight_layer_{key}", np.mean(metrics["critic"]["weights"][key])
                )
                self.model.logger.record(
                    f"critic_grad_layer_{key}", np.mean(metrics["critic"]["gradient"][key])
                )

            # 5. Parâmetros dinâmicos do PPO
            if hasattr(self.model.policy, "log_std"):
                self.model.logger.record(
                    "policy_log_std", torch.exp(self.model.policy.log_std).mean().item()
                )

        # Armazena o snapshot fiel no buffer interno
        self.eval_logs.append(self.model.logger.name_to_value.copy())

    def collect_reward_metrics(self):
        """Coleta as recompensas de cada episódio individualmente na vertical."""
        buffer = getattr(self.model, "ep_info_buffer", None)

        if buffer is not None and len(buffer) > 0:
            for ep_info in buffer:
                # Damos append direto no float. Cada episódio vira um elemento isolado.
                self.reward_logs.append(ep_info["r"])
            buffer.clear()

    def flush_logs_to_disk(self):
        """Descarrega o acumulador de memória para o arquivo CSV físico."""
        if self.eval_logs:
            df = pd.DataFrame(self.eval_logs)
            df.to_csv(
                self.metrics_directory,
                mode="a" if os.path.exists(self.metrics_directory) else "w",
                index=False,
                header=not os.path.exists(self.metrics_directory),
            )
            self.eval_logs.clear()

        if self.reward_logs:
            # Criamos o DataFrame explicitando que a lista é uma única coluna vertical
            df_rewards = pd.DataFrame(self.reward_logs, columns=["recompensa_episodio"])
            df_rewards.to_csv(
                self.rewards_directory,
                mode="a" if os.path.exists(self.rewards_directory) else "w",
                index=False,
                header=not os.path.exists(self.rewards_directory),
            )
            self.reward_logs.clear()  # Limpa a memória para os próximos episódios
