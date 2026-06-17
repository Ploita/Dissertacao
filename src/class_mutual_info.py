import copy

import npeet.entropy_estimators as ee
import torch

from class_fetcher import ActivationFetcher


class MutualInfoCalculator:
    """Isola a lógica combinatória e o cálculo estatístico de MI."""
    def __init__(self, model, layer_size: int, has_reference: bool):
        self.model = model
        self.has_reference = has_reference

        layer_names_raw = [f'h_{i+1}' for i in range(layer_size)]
        layer_names_raw.insert(0, 'X')
        output_names_raw = copy.copy(layer_names_raw)
        output_names_raw.extend(['hat Y'])

        if self.has_reference:
            output_names_raw.extend(['Y'])

        def get_safe_name(name: str) -> str:
            if name == 'Y':
                return 'Y_ref'
            return name.replace('h_', 'h').replace('hat Y', 'Yhat')

        raw_to_safe_map = {raw: get_safe_name(raw) for raw in output_names_raw}

        self.mapping = {}
        for i, j_raw in enumerate(layer_names_raw):
            for k_raw in output_names_raw[i+1:]:
                j_safe = raw_to_safe_map.get(j_raw, j_raw)
                k_safe = raw_to_safe_map.get(k_raw, k_raw)
                key_safe = f"I_{j_safe}_{k_safe}"
                self.mapping[key_safe] = (j_raw, k_raw)

    def compute(self, metrics: dict, fetcher: ActivationFetcher, observations: torch.Tensor):
        """Injeta os extremos de dados controlados e computa as métricas do npeet."""
        with torch.no_grad():
            features = self.model.policy.extract_features(observations)
            features_np = features.cpu().numpy()

            # Injeção controlada das bordas (X, hat Y, Y)
            fetcher.activations["actor_X"] = features_np
            fetcher.activations["critic_X"] = features_np

            fetcher.activations["actor_hat Y"] = self.model.policy.action_net(
                self.model.policy.mlp_extractor.policy_net(features)
            ).cpu().numpy()
            fetcher.activations["critic_hat Y"] = self.model.policy.value_net(
                self.model.policy.mlp_extractor.value_net(features)
            ).cpu().numpy()

            if self.has_reference and self.model.reference_agent is not None:
                ref_out = self.model.reference_agent.predict(observations)[0]
                ref_out_tensor = torch.from_numpy(ref_out).to(self.model.device)
                fetcher.activations["actor_Y"] = ref_out_tensor.cpu().numpy()
                fetcher.activations["critic_Y"] = ref_out_tensor.cpu().numpy()

        # Execução do cálculo matemático par a par
        for key_safe, (raw1, raw2) in self.mapping.items():
            metrics['actor']['mutual_info'][key_safe].append(ee.mi(
                fetcher.activations[f"actor_{raw1}"],
                fetcher.activations[f"actor_{raw2}"]
            ))
            metrics['critic']['mutual_info'][key_safe].append(ee.mi(
                fetcher.activations[f"critic_{raw1}"],
                fetcher.activations[f"critic_{raw2}"]
            ))
