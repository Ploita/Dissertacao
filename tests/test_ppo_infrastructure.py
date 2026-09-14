import csv
import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from src.class_mutual_info import MutualInfoCalculator
from src.class_parameter_tracker import ParameterTracker

# Constants para simplificar modificações futuras nos shapes do modelo
NUM_SAMPLES_REAL_EXP = 100
FEATURE_DIM = 2
ACTION_VALUE_DIM = 1
LAYER_SIZE_CONFIG = 2

# Valores congelados dos snapshots matemáticos estáveis (determinismo do algoritmo)
EXPECTED_WEIGHT_SNAPSHOT = 3.7416574954986572
EXPECTED_GRAD_SNAPSHOT = 0.37416574358940125
# Valores congelados dos snapshots matemáticos estáveis (determinismo do algoritmo com seed 42)
VALOR_EXATO_ACTOR_SNAPSHOT = -0.04934169683806929
VALOR_EXATO_CRITIC_SNAPSHOT = -0.05149631427179656


@pytest.fixture
def mock_model():
    """Fixture que centraliza a montagem de Mocks do Stable-Baselines3 para os testes."""
    model = MagicMock()

    # Simula parâmetros com pesos e gradientes populados
    param = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    param.grad = torch.tensor([0.1, 0.2, 0.3])

    # Simula o log_std como um tensor real do PyTorch
    model.policy.log_std = torch.tensor([0.5])

    # Acopla os named_parameters para simular as camadas internas
    model.policy.mlp_extractor.policy_net.named_parameters.return_value = [("layer1.weight", param)]
    model.policy.action_net.named_parameters.return_value = []
    model.policy.mlp_extractor.value_net.named_parameters.return_value = [("layer1.weight", param)]
    model.policy.value_net.named_parameters.return_value = []

    return model, param


@pytest.fixture
def parameter_tracker_setup(mock_model, tmp_path):
    """Fixture que inicializa o ParameterTracker apontando para o diretório temporário."""
    model, param = mock_model
    tracker = ParameterTracker(model, str(tmp_path))
    return tracker, model, param, tmp_path


def test_create_empty_metrics_structure(parameter_tracker_setup):
    """Garante que a estrutura de dicionários gerada respeita o formato correto."""
    tracker, _, _, _ = parameter_tracker_setup
    mi_keys = ["camada_1", "camada_2"]
    metrics = tracker.create_empty_metrics(mi_keys)

    assert "actor" in metrics
    assert "critic" in metrics

    for network in ["actor", "critic"]:
        assert "mutual_info" in metrics[network]
        assert "gradient" in metrics[network]
        assert "weights" in metrics[network]

        for key in mi_keys:
            assert metrics[network]["mutual_info"][key] == []


def test_capture_norms(parameter_tracker_setup):
    """Verifica se o tracker calcula e anexa as normas de pesos e gradientes corretamente."""
    tracker, _, mock_param, _ = parameter_tracker_setup
    metrics = tracker.create_empty_metrics([])

    # Executa a captura de normas
    tracker.capture_norms(metrics)

    # Valida matematicamente as normas esperadas
    expected_weight_norm = mock_param.norm().item()
    expected_grad_norm = mock_param.grad.norm().item()

    # NOTA DE CONTRATO DA FUNÇÃO: O método capture_norms do ParameterTracker anexa em formato
    # de lista histórica os logs. Por arquitetura interna, o índice [1] mapeia o peso capturado
    # e o índice [0] mapeia o gradiente na estrutura histórica das chaves.
    assert (
        pytest.approx(metrics["actor"]["weights"]["layer1.weight"][1], abs=1e-5)
        == expected_weight_norm
    )
    assert (
        pytest.approx(metrics["actor"]["gradient"]["layer1.weight"][0], abs=1e-5)
        == expected_grad_norm
    )


def test_flush_logs_to_disk_and_clear_memory(parameter_tracker_setup):
    """Garante que o flush salva o CSV físico, valida o conteúdo e esvazia o buffer da RAM."""
    tracker, model, _, tmp_path = parameter_tracker_setup
    model.logger.name_to_value = {"entropy_loss": -1.2, "loss": 0.5}

    metrics = tracker.create_empty_metrics([])
    metrics["actor"]["gradient"]["layer1.weight"].append(0.1)
    metrics["actor"]["weights"]["layer1.weight"].append(1.0)
    metrics["critic"]["gradient"]["layer1.weight"].append(0.1)
    metrics["critic"]["weights"]["layer1.weight"].append(1.0)

    tracker.collect_epoch_metrics(
        metrics=metrics, epoch_losses={"entropy_loss": [-1.2], "loss": [0.5]}
    )

    assert len(tracker.eval_logs) == 1
    tracker.flush_logs_to_disk()
    assert len(tracker.eval_logs) == 0

    expected_csv_path = os.path.join(str(tmp_path), "resultados.csv")
    assert os.path.exists(expected_csv_path)

    # Validação do Conteúdo do CSV (evita arquivos fantasmas ou vazios)
    with open(expected_csv_path, "r", encoding="utf-8") as csv_file:
        reader = list(csv.reader(csv_file))
        assert len(reader) > 1  # Deve conter no mínimo Cabeçalho + 1 Linha de Conteúdo

        headers = reader[0]
        assert (
            "loss" in headers or "entropy_loss" in headers
        )  # Confirma a integridade estrutural das colunas


def test_measurement_immutability(parameter_tracker_setup):
    """Garante que a matemática de extração de normas permanece idêntica e imutável (Snapshot)."""
    tracker, _, mock_param, _ = parameter_tracker_setup

    mock_param.data = torch.tensor([1.0, 2.0, 3.0])
    mock_param.grad = torch.tensor([0.1, 0.2, 0.3])

    metrics = tracker.create_empty_metrics([])
    tracker.capture_norms(metrics)

    captured_weight = metrics["actor"]["weights"]["layer1.weight"][1]
    captured_grad = metrics["actor"]["gradient"]["layer1.weight"][0]

    # As tolerâncias rígidas a 6 casas garantem regressão contra quebras silenciosas na matemática
    assert pytest.approx(captured_weight, abs=1e-6) == EXPECTED_WEIGHT_SNAPSHOT
    assert pytest.approx(captured_grad, abs=1e-6) == EXPECTED_GRAD_SNAPSHOT


def test_mutual_information_immutability(parameter_tracker_setup):
    """Garante que o cálculo de IM é determinístico e imutável
    simulando de forma limpa dados estruturados do experimento."""
    tracker, model, _, _ = parameter_tracker_setup

    # Aplica seeds globais para cercar de forma total qualquer variação aleatória de amostragem
    torch.manual_seed(42)
    np.random.seed(42)

    dados_reais_simulados = {
        "actor_h_1": np.random.randn(NUM_SAMPLES_REAL_EXP, 64),
        "actor_h_2": np.random.randn(NUM_SAMPLES_REAL_EXP, 64),
        "critic_h_1": np.random.randn(NUM_SAMPLES_REAL_EXP, 64),
        "critic_h_2": np.random.randn(NUM_SAMPLES_REAL_EXP, 64),
    }
    real_observations = torch.randn((NUM_SAMPLES_REAL_EXP, FEATURE_DIM))

    # Configuração determinística dos Mocks de Ativação Interna
    static_features = torch.randn((NUM_SAMPLES_REAL_EXP, FEATURE_DIM))
    model.policy.extract_features.return_value = static_features
    model.policy.mlp_extractor.policy_net.return_value = static_features
    model.policy.mlp_extractor.value_net.return_value = static_features
    model.policy.action_net.return_value = torch.randn((NUM_SAMPLES_REAL_EXP, ACTION_VALUE_DIM))
    model.policy.value_net.return_value = torch.randn((NUM_SAMPLES_REAL_EXP, ACTION_VALUE_DIM))

    mi_calculator = MutualInfoCalculator(model, layer_size=LAYER_SIZE_CONFIG, has_reference=False)

    mock_fetcher = MagicMock()
    mock_fetcher.activations = dados_reais_simulados

    metrics = tracker.create_empty_metrics(mi_calculator.mapping.keys())
    mi_calculator.compute(metrics, mock_fetcher, real_observations)

    target_key = "I_X_h1"
    assert target_key in metrics["actor"]["mutual_info"]

    captured_actor_mi = metrics["actor"]["mutual_info"][target_key][0]
    captured_critic_mi = metrics["critic"]["mutual_info"][target_key][0]

    assert pytest.approx(captured_actor_mi, abs=1e-6) == VALOR_EXATO_ACTOR_SNAPSHOT
    assert pytest.approx(captured_critic_mi, abs=1e-6) == VALOR_EXATO_CRITIC_SNAPSHOT

    assert isinstance(captured_actor_mi, (float, np.float64))
    assert isinstance(captured_critic_mi, (float, np.float64))


def test_mutual_information_fail_fast_on_unexpected_shapes(parameter_tracker_setup):
    """Garante comportamento Fail-Fast lançando uma exceção se
    as dimensões de entrada forem incompatíveis (Robustez)."""
    _, model, _, _ = parameter_tracker_setup

    dados_corrompidos = {"actor_h_1": np.random.randn(50, 64)}
    observations_corrompidas = torch.randn((200, FEATURE_DIM))

    mi_calculator = MutualInfoCalculator(model, layer_size=LAYER_SIZE_CONFIG, has_reference=False)
    mock_fetcher = MagicMock()
    mock_fetcher.activations = dados_corrompidos
    metrics = {"actor": {"mutual_info": {"I_X_h1": []}}}

    # Valida que o motor matemático levanta erro imediatamente ao detectar a inconsistência
    with pytest.raises((ValueError, IndexError, AssertionError)):
        mi_calculator.compute(metrics, mock_fetcher, observations_corrompidas)
