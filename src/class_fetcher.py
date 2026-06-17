class ActivationFetcher:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.handles = []

    def attach_hooks(self):
        """Registra os hooks dinamicamente no início de cada ciclo do train()."""
        self.remove()

        networks = {
            "actor": self.model.policy.mlp_extractor.policy_net,
            "critic": self.model.policy.mlp_extractor.value_net
        }

        # Limpeza preventiva em baixo nível no PyTorch antes do re-atracamento
        for net in networks.values():
            for module in net:
                if hasattr(module, '_forward_hooks'):
                    module._forward_hooks.clear()

        for net_name, net in networks.items():
            layer_idx = 1
            for i, module in enumerate(net):
                if i % 2 == 1:
                    hook_name = f"{net_name}_h_{layer_idx}"
                    handle = module.register_forward_hook(self.get_hook(hook_name))
                    self.handles.append(handle)
                    layer_idx += 1

    def get_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu().numpy()
        return hook

    def clear(self):
        """Limpa o dicionário garantindo que nenhuma referência antiga resista."""
        self.activations = {}

    def remove(self):
        """Remove os hooks atuais do Grafo."""
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()
        self.activations.clear()


