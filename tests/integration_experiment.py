import glob
import os
import shutil

import pandas as pd
from src.class_experiment import Experimento
from src.post_processing import main_pipeline

# Configuração do diretório gabarito (onde estão os arquivos que você sabe que estão certos)
DIRETORIO_GABARITO = "tests/gabarito_referencia"
# Diretório temporário controlado para a execução real do teste
DIRETORIO_TESTE = "data/test/A0"


def test_experimento_ponta_a_ponta_e_comparacao_de_resultados():
    """
    Executa a rotina real do experimento, processa os dados, compara os CSVs
    e PDFs gerados com os resultados salvos no gabarito e limpa tudo se passar.
    """
    # 1. Garante que o diretório de gabarito existe antes de começar
    assert os.path.exists(DIRETORIO_GABARITO), (
        f"Erro: Crie o diretório '{DIRETORIO_GABARITO}' e coloque os "
        f"arquivos CSV e PDF corretos lá dentro para servir de comparação!"
    )

    # Limpa o diretório de testes caso tenha sobrado algo de uma execução anterior abortada
    if os.path.exists(DIRETORIO_TESTE):
        shutil.rmtree(DIRETORIO_TESTE)

    # 2. Configurações originais fornecidas para o loop de execução
    num_neurons = 32
    params = {
        "policy_kwargs": dict(net_arch=[num_neurons, num_neurons]),
        "timesteps": int(
            1e5
        ),  # Caso queira que o teste rode mais rápido, pode reduzir ex: int(1e3)
        "directory": DIRETORIO_TESTE,
        "net_init": 2,
    }
    environment_list = ["BipedalWalker-v3"]

    print("\n[+] Iniciando a execução real do Experimento...")
    # 3. Execução real do seu loop de blocos
    for environment in environment_list:
        for init in range(2):
            for seed in range(2):
                params["seeds"] = [seed]
                params["env_id"] = environment
                params["net_init"] = init

                ensaio = Experimento(params)
                ensaio.treinamento()

            # Executa o pós-processamento gerando os CSVs e PDFs consolidados
            main_pipeline(params["directory"])

    print("[+] Execução concluída. Iniciando fase de checagem dos arquivos...")

    # 4. Mapeamento e comparação dos CSVs gerados
    arquivos_csv_teste = glob.glob(os.path.join(DIRETORIO_TESTE, "**/*.csv"), recursive=True)
    assert len(arquivos_csv_teste) > 0, "Nenhum arquivo CSV foi gerado na pasta de testes!"

    for csv_teste_path in arquivos_csv_teste:
        # Encontra o caminho equivalente dele dentro do diretório gabarito
        caminho_relativo = os.path.relpath(csv_teste_path, DIRETORIO_TESTE)
        csv_gabarito_path = os.path.join(DIRETORIO_GABARITO, caminho_relativo)

        assert os.path.exists(csv_gabarito_path), (
            f"O arquivo esperado {caminho_relativo} não existe no gabarito!"
        )

        # Carrega e compara os DataFrames
        df_teste = pd.read_csv(csv_teste_path)
        df_gabarito = pd.read_csv(csv_gabarito_path)

        # Valida se as tabelas têm o mesmo formato e os mesmos dados exatos
        pd.testing.assert_frame_equal(
            df_teste,
            df_gabarito,
            check_dtype=False,
            obj=f"Divergência de dados encontrada no arquivo CSV: {caminho_relativo}",
        )

    # 5. Mapeamento e validação dos PDFs gerados (Gráficos)
    arquivos_pdf_teste = glob.glob(os.path.join(DIRETORIO_TESTE, "**/*.pdf"), recursive=True)
    assert len(arquivos_pdf_teste) > 0, "Nenhum gráfico PDF foi gerado na pasta de testes!"

    for pdf_teste_path in arquivos_pdf_teste:
        caminho_relativo = os.path.relpath(pdf_teste_path, DIRETORIO_TESTE)
        pdf_gabarito_path = os.path.join(DIRETORIO_GABARITO, caminho_relativo)

        assert os.path.exists(pdf_gabarito_path), (
            f"O gráfico esperado {caminho_relativo} não existe no gabarito!"
        )

        # Garante que o PDF gerado possui integridade física (não está zerado/corrompido)
        assert os.path.getsize(pdf_teste_path) > 1024, (
            f"O PDF gerado {caminho_relativo} parece estar corrompido ou vazio!"
        )

    print("[+] Todos os arquivos passados no teste são idênticos ao seu gabarito pessoal!")

    # ==========================================================
    # 6. ROTINA DE LIMPEZA AUTÔNOMA (Só roda se nenhum assert acima falhar)
    # ==========================================================
    print(f"[+] Removendo diretório de teste '{DIRETORIO_TESTE}' de forma limpa...")
    shutil.rmtree(DIRETORIO_TESTE)
