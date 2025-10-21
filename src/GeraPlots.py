from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

plt.style.use('style.mplstyle')

def apply_latex_legend(col_name, type):
    
    latex_label = f'\\text{{{col_name}}}'

    if type == 'loss':
        latex_base = col_name.removesuffix('_loss')
        latex_base = latex_base.replace('_', ' ')
        latex_label = f'$\\mathcal{{L}}_{{{latex_base}}}$'
        
    else:
        match = re.search(r'(actor|critic)[._\s]?(weight|grad)[._\s]?layer(?:[._\s](\d+))?.(weight|bias)', col_name)
        if match:
            prefix = match.group(2)
            layer = match.group(3)
            symbol = match.group(4)

            latex_symbol_prefix = f'\\nabla' if prefix=='grad' else ''
            latex_symbol = 'W' if symbol=='weight' else 'b'
            
            # Se não houver número de camada, assume-se que é a camada de saída (Out).
            layer_base = int(layer)//2 if layer is not None else '\\text{Out}'
            
            # Base: W_i/W_Out ou b_i/b_Out
            latex_label = f'${latex_symbol_prefix}\\text{{{latex_symbol}}}_{{{layer_base}}}$'
    
    return latex_label

def combinar_strings(tupla_de_chaves_seguras: tuple[str, str]) -> str:
    """
    Combina duas chaves seguras do tipo I_A_B e I_B_C em uma chave de plot I_A_B_C.
    
    Ex: ('I_X_h1', 'I_h1_h2') -> 'X_h1_h2' (chave segura para nome de arquivo)
    """
    primeira = tupla_de_chaves_seguras[0] # Ex: 'I_X_h1'
    segunda = tupla_de_chaves_seguras[1] # Ex: 'I_h1_h2'
    
    # Remove o 'I_' do começo
    args_primeira_str = primeira[2:] # Ex: 'X_h1'
    args_segunda_str = segunda[2:]   # Ex: 'h1_h2'
    
    lista_args = args_primeira_str.split('_') + args_segunda_str.split('_')
    args_combinados_unicos = list(dict.fromkeys(lista_args))
    
    # Retorna o nome seguro combinado: X_h1_h2
    return f'{"_".join(args_combinados_unicos)}'

def gera_combinacoes(colunas_mi_seguras: list) -> list[tuple[str, str]]:
    """
    Gera combinações sequenciais de MI (ex: I_A_B e I_B_C).
    
    A entrada agora é uma lista de strings de chaves seguras (ex: ['actor_I_X_h1', 'actor_I_h1_h2', ...]).
    A saída são tuplas de chaves seguras, mas sem o prefixo (ex: ('I_X_h1', 'I_h1_h2')).
    """
    combinacoes_sequenciais = []
    
    # Mapeia a chave de MI segura (sem prefixo) para seus componentes (A, B)
    stripped_info = {}
    
    # 1. Extrair os componentes (A, B) de cada chave (I_A_B)
    for col_full in colunas_mi_seguras:
        # Remove o prefixo 'actor_' ou 'critic_'
        col_safe = col_full.split('_', 1)[1].split('_mean')[0] # Ex: 'I_X_h1'
        
        # O split pega a primeira letra 'I' e os dois componentes 'X' e 'h1'
        parts = col_safe.split('_')
        if len(parts) == 3 and parts[0] == 'I':
            stripped_info[col_safe] = [parts[1], parts[2]] # Ex: ['X', 'h1']

    # 2. Encontrar a sequência I_A_B e I_B_C
    for col1_safe, info1 in stripped_info.items():
        if len(info1) == 2:  # Deve ser I_A_B
            segundo_elemento_col1 = info1[1] # O 'B'
            for col2_safe, info2 in stripped_info.items():
                if col1_safe != col2_safe and len(info2) == 2:
                    primeiro_elemento_col2 = info2[0] # O 'B'
                    
                    # Encontrou a sequência: I_A_B seguido de I_B_C
                    if segundo_elemento_col1 == primeiro_elemento_col2:
                        combinacoes_sequenciais.append((col1_safe, col2_safe))
                        
    return combinacoes_sequenciais

def fechar_plot(directory, filename, xlabel='Época', ylabel='Valor'):
    """Salva o gráfico atual e o fecha, garantindo o tight layout."""
    plots_dir = os.path.join(directory, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{filename}.pdf'))
    plt.close()

def coletar_dados_experimentos(caminho_base):
    """
    Varre as subpastas, lê 'resultados.csv', concatena e salva o arquivo compilado.
    Retorna o DataFrame compilado.
    """
    dados_totais = []
    
    try:
        conteudo_pasta = os.listdir(caminho_base)
    except FileNotFoundError:
        print(f"Erro: O caminho base '{caminho_base}' não foi encontrado.")
        return None    
    
    for nome_pasta in conteudo_pasta:
        caminho_experimento = os.path.join(caminho_base, nome_pasta)

        if os.path.isdir(caminho_experimento):
            caminho_csv = os.path.join(caminho_experimento, 'resultados.csv')
            if os.path.exists(caminho_csv):
                try:
                    df = pd.read_csv(caminho_csv)
                    df['id_experimento'] = nome_pasta
                    dados_totais.append(df)
                except Exception as e:
                    print(f"Erro ao ler o arquivo {caminho_csv}: {e}")
            else:
                pass 

    if not dados_totais:
        print("Nenhum dado de experimento ('resultados.csv') foi encontrado nas subpastas.")
        return None

    df_final = pd.concat(dados_totais, ignore_index=True)
    
    # Cria o nome do arquivo de saída usando o nome do ambiente
    match = re.search(r'groups/([^/]+)', caminho_base)
    ambiente_nome = match.group(1) if match else 'unknown_environment'
    
    caminho_saida = os.path.join(caminho_base, 'plots', f"compilado_{ambiente_nome}.csv")
    
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    df_final.to_csv(path_or_buf=caminho_saida, mode='w', index=False)
    return df_final

def load_and_group_data(df):
    """
    Recebe o DataFrame compilado, calcula o 'timestep' por experimento, 
    e então calcula Média e Desvio Padrão para TODAS as métricas.
    """
    if df is None or df.empty:
        return None

    if 'id_experimento' not in df.columns:
        print(f"Erro: Coluna 'id_experimento' não encontrada no DataFrame compilado.")
        return None

    # Garante que 'id_experimento' não seja agrupada
    group_by_cols = [col for col in df.columns if col != 'id_experimento']
    
    # Cria a coluna 'timestep' (época/iteração) para cada experimento
    df['timestep'] = df.groupby('id_experimento').cumcount()
    
    # Agrupa por 'timestep' e calcula a Média e o STD de todas as métricas
    grouped_data = df.groupby('timestep')[group_by_cols].agg(['mean', 'std'])
    
    # Renomeia as colunas para o formato 'métrica_agregação'
    grouped_data.columns = [f'{col[0]}_{col[1]}' for col in grouped_data.columns]
    
    return grouped_data.reset_index()

def load_rewards_from_folders(root_directory, folder_list, rewards_filename='rewards.csv'):
    """
    Lê os rewards.csv, transforma (transpõe) e calcula a Média e STD por timestep.
    """
    all_experiments_df = []
    
    for folder_name in folder_list:
        file_path = os.path.join(root_directory, folder_name, rewards_filename)
        if not os.path.exists(file_path):
            continue
        try:
            # Lê, transpõe e garante que os dados são numéricos
            df_reward = pd.read_csv(file_path, header=0).T
            df_reward = df_reward.apply(pd.to_numeric, errors='coerce')
            df_reward['experiment_id'] = folder_name
            all_experiments_df.append(df_reward)
            
        except Exception as e:
            print(f"Erro ao processar {file_path}: {e}")
            
    if not all_experiments_df:
        return None

    combined_df = pd.concat(all_experiments_df, ignore_index=True)
    data_cols = combined_df.drop(columns=['experiment_id']).dropna(axis=1, how='all')
    
    mean_rewards = data_cols.mean(axis=0)
    std_rewards = data_cols.std(axis=0)
    
    grouped_data = pd.DataFrame({
        'timestep': mean_rewards.index, 
        'mean_reward': mean_rewards.values,
        'std_reward': std_rewards.values
    })
    
    grouped_data['timestep'] = pd.to_numeric(grouped_data['timestep'], errors='coerce')

    return grouped_data

def plot_reward_mean_std(grouped_data, directory):
    """Gera o plot de Recompensa (Média +/- std)."""
    if grouped_data is None or grouped_data.empty:
        print("Aviso: Dados de recompensa agrupados estão vazios. Pulando plotagem de Recompensa.")
        return
        
    plt.figure()
    mean = grouped_data['mean_reward']
    std = grouped_data['std_reward']
    timestep = grouped_data['timestep']
    
    plt.plot(timestep, mean, label='Média de Recompensa', color='blue')
    
    plt.fill_between(
        timestep,
        mean - std, 
        mean + std,   
        alpha=0.2,                     
        color='blue',
    )

    fechar_plot(directory, 'reward_mean_std', 'Iteração', 'Recompensa Média')

def plot_rl_metrics(data_grouped, directory):
    """
    Gera plots de Loss (com componentes), Gradiente e Pesos (média +/- std).
    MODIFICADO: Usa a função _get_latex_label para formatar as legendas.
    """
    if data_grouped is None or data_grouped.empty:
        print("DataFrame agrupado vazio ou None, pulando a plotagem.")
        return
        
    # Definindo os mapeamentos de métricas, incluindo o tipo para a lógica LaTeX
    metric_map = OrderedDict([
        ('loss', {'title': 'Componentes da Loss', 'ylabel': 'Valor da Loss Média', 'components': ['entropy_loss', 'policy_gradient_loss', 'value_loss'], 'type': 'loss'}),
        ('actor_weight', {'title': 'Pesos do Ator', 'ylabel': 'Valor do Peso', 'components': None, 'type': 'weight'}),
        ('actor_grad', {'title': 'Gradiente do Ator', 'ylabel': 'Valor do Gradiente', 'components': None, 'type': 'grad'}),
        ('critic_weight', {'title': 'Pesos do Crítico', 'ylabel': 'Valor do Peso', 'components': None, 'type': 'weight'}),
        ('critic_grad', {'title': 'Gradiente do Crítico', 'ylabel': 'Valor do Gradiente', 'components': None, 'type': 'grad'}),
    ])

    for metric_prefix, info in metric_map.items():
        plt.figure(figsize=(10, 6))
        
        plot_type = info['type']
        
        if info['components']:
            # Caso de Loss: Os nomes são a base (e.g., 'entropy_loss')
            plot_list = info['components']
        else:
            # Caso de Peso/Gradiente: Encontra as colunas base (sem _mean/_std)
            # Remove o sufixo '_mean' para obter o nome base do parâmetro
            all_means = [col.removesuffix('_mean') for col in data_grouped.columns 
                         if col.startswith(metric_prefix) and col.endswith('_mean')]
            if not all_means:
                continue
            plot_list = all_means

        for prefix in plot_list:
            mean_col = f'{prefix}_mean'
            std_col = f'{prefix}_std'
            
            if mean_col not in data_grouped.columns:
                continue

            legend_name = apply_latex_legend(prefix, plot_type)
            # ---------------------------

            # Plot da linha principal (média)
            plt.plot(data_grouped['timestep'], data_grouped[mean_col], label=legend_name)
            
            if std_col in data_grouped.columns:
                # Adiciona a área de fill (média +/- std)
                plt.fill_between(
                    data_grouped['timestep'],
                    data_grouped[mean_col] - data_grouped[std_col],
                    data_grouped[mean_col] + data_grouped[std_col],
                    alpha=0.2,
                )
        
        fechar_plot(directory, metric_prefix, 'Época', info['ylabel'])

def plot_im(data: pd.DataFrame, directory: str):
    """
    Gera scatter plots para as combinações sequenciais de Informação Mútua.
    Utiliza as colunas '_mean' para os valores dos eixos.
    """
    
    # FILTRA APENAS AS COLUNAS DE MÉDIA para garantir que a função gera_combinacoes
    # receba uma lista consistente de colunas MI (sem duplicar por mean/std).
    actor_mi_mean_cols = data.filter(regex=r'^actor_I_.+_mean$').columns.tolist()
    
    # gera_combinacoes extrai as chaves SEGURAS (ex: 'I_X_h1') a partir das colunas de MI.
    # Ex: a partir de ['actor_I_X_h1_mean', 'actor_I_h1_h2_mean', ...], retorna [('I_X_h1', 'I_h1_h2'), ...]
    combinacoes_sequenciais_seguras = gera_combinacoes(colunas_mi_seguras=actor_mi_mean_cols) 

    # Definir a coluna de cor baseada no índice
    size = len(data)
    color_col = np.arange(0, size)

    for prefix in ['actor', 'critic']:
        # Filtra todas as colunas de MI (mean e std) para o prefixo atual
        mi_data = data.filter(regex=rf'^{prefix}_I_.+')
        
        # Percorre as combinações sequenciais (as chaves SEGURAS sem o prefixo)
        for col1_safe, col2_safe in combinacoes_sequenciais_seguras:
            # Recompõe os nomes completos das colunas NO DATAFRAME, **ADICIONANDO O SUFIXO _mean**
            col1_full_name = f'{prefix}_{col1_safe}_mean'
            col2_full_name = f'{prefix}_{col2_safe}_mean'

            col1_std = f'{prefix}_{col1_safe}_std'
            col2_std = f'{prefix}_{col2_safe}_std'
            
            # Verifica se as colunas de MÉDIA existem no DataFrame
            if col1_full_name not in mi_data.columns or col2_full_name not in mi_data.columns:
                continue
                        
            # VAMOS USAR APENAS AS COLUNAS DE MÉDIA para o scatter plot
            val1 = mi_data[col1_full_name]
            val2 = mi_data[col2_full_name]
                    
            plt.figure()
            plt.scatter(val1, val2, c=color_col, cmap='magma', s= (8 + np.mean(mi_data[[col1_std, col2_std]]))**2 )
            
           # Cria a chave segura combinada para o nome do arquivo
            combined_safe_key = combinar_strings((col1_safe, col2_safe))
            
            # Função auxiliar para converter o nome seguro (X, h1, Yhat) para LaTeX (X, h_1, \hat{Y})
            def safe_to_latex(safe_key: str) -> str:
                key = safe_key.replace('I_', '').replace('_', ',')
                # Troca h1 por h_1, h2 por h_2, etc. (se houver mais de um dígito, o regex segura)
                key = re.sub(r'(h)(\d+)', r'h_{\2}', key)
                key = key.replace('Yhat', '\\hat{Y}')
                key = key.replace('Y_ref', 'Y') # Y_ref deve ser Y no plot
                return f'I({key})'
            
            # Labels dos eixos (ex: $I(X,h_1)$)
            col_x_label_latex = f'${safe_to_latex(col1_safe)}$'
            col_y_label_latex = f'${safe_to_latex(col2_safe)}$'
            
            # Nome do arquivo (usa a chave segura combinada)
            fechar_plot(f'{directory}', f'{prefix}_{combined_safe_key}', col_x_label_latex, col_y_label_latex)


def main_pipeline(root_directory):
    """
    Executa o pipeline completo: coleta, agrupa e plota.
    """
    try:
        df_completo = coletar_dados_experimentos(root_directory)
    except TypeError:
         # Se a função retornar None, o df_completo será None, e o pipeline para aqui.
         print("Erro na coleta de dados. Terminando o pipeline.")
         return
    
    if df_completo is None:
        return
        
    output_dir = root_directory
    grouped_data_compiled = load_and_group_data(df_completo)
    
    if grouped_data_compiled is not None:
        
        # Plotagem de Loss, Gradiente e Pesos
        try:
            plot_rl_metrics(grouped_data_compiled, output_dir)
        except Exception as e:
            print(f"Ocorreu um erro ao gerar plots de métricas RL: {e}")
            
        # Plotagem de Informação Mútua (IM)
        try:
            plot_im(grouped_data_compiled, output_dir)
        except Exception as e:
            print(f"Ocorreu um erro ao gerar plots de IM: {e}")
            
    try:
        folder_names = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
        if folder_names:
            grouped_rewards = load_rewards_from_folders(root_directory, folder_names)
            if grouped_rewards is not None:
                plot_reward_mean_std(grouped_rewards, output_dir)
        else:
             print(f"\nAviso: Nenhuma subpasta de experimento encontrada em {root_directory}. Pulando plotagem de Recompensa.")
    except FileNotFoundError:
        print(f"\nAviso: Diretório de recompensas {root_directory} não encontrado. Pulando plotagem de Recompensa.")
    except Exception as e:
        print(f"\nOcorreu um erro ao processar plots de Recompensa: {e}")