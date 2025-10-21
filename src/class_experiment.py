from stable_baselines3.common.monitor import Monitor
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import gymnasium
import torch 
import json
import os
import re

from class_ppo import PPO_tunado
plt.style.use('style.mplstyle')

def criar_pasta(directory: str) -> str:
    """
    Cria uma nova pasta numerada sequencialmente no diretório dado.
    """
    os.makedirs(directory, exist_ok=True)
    max_number = 0
    pattern = re.compile(r'^(\d{3})$')

    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)) and pattern.match(item):
            try:
                current_number = int(item)
                if current_number > max_number:
                    max_number = current_number
            except ValueError:
                pass

    next_number = max_number + 1
    new_folder_name = str(next_number).zfill(3)
    new_directory = os.path.join(directory, new_folder_name)

    os.makedirs(new_directory)
    return new_directory

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
        col_safe = col_full.split('_', 1)[1] # Ex: 'I_X_h1'
        
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

def combinar_strings(tupla_de_chaves_seguras: tuple[str, str]) -> str:
    """
    Combina duas chaves seguras do tipo I_A_B e I_B_C em uma chave de plot I_A_B_C.
    
    Ex: ('I_X_h1', 'I_h1_h2') -> 'I_X_h1_h2' (chave segura para nome de arquivo)
    """
    primeira = tupla_de_chaves_seguras[0] # Ex: 'I_X_h1'
    segunda = tupla_de_chaves_seguras[1] # Ex: 'I_h1_h2'
    
    # Remove o 'I_' do começo
    args_primeira_str = primeira[2:] # Ex: 'X_h1'
    args_segunda_str = segunda[2:]   # Ex: 'h1_h2'
    
    lista_args = args_primeira_str.split('_') + args_segunda_str.split('_')
    # Mantém apenas os elementos únicos na ordem de ocorrência (X, h1, h2)
    args_combinados_unicos = list(dict.fromkeys(lista_args))
    
    # Retorna o nome seguro combinado: I_X_h1_h2
    return f'{"_".join(args_combinados_unicos)}'


def fechar_plot(directory, plot_name, axle_x = 'Época', axle_y = 'Valor'):
    """Salva o plot em formato PDF e fecha."""
    plt.xlabel(axle_x)
    plt.ylabel(axle_y)
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'{directory}/plots/{plot_name}.pdf')
    plt.close()

class Experimento():
    def __init__(self, params) -> None:
        # Setagem
        self.env_id = 'CartPole-v1'
        self.n_envs = 4
        self.policy_kwargs = dict(net_arch = [32,32])
        self.seeds = [0]
        self.timesteps = int(1e3) 
        self.reference_agent = None
        self.calc_mutual_info = True
        self.directory = '../data/results'

        # Model Parameters
        self.learning_rate = 3e-4
        self.n_steps = 2048
        self.batch_size = 64
        self.n_epochs = 10
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.clip_range_vf = None
        self.normalize_advantage = True
        self.ent_coef = 0.0
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.use_sde = False
        self.sde_sample_freq = -1
        self.rollout_buffer_class = None
        self.rollout_buffer_kwargs = None
        self.target_kl = None
        self.stats_window_size = 100
        # self.tensorboard_log =  "../data/tensorboard_logs/"
        self.verbose = 0
        self.device = "cpu"
        
        # Recording Parameters
        self.recording = False
        self.recording_ep_freq = 100
        self.coleta = False
        # Fim da setagem

        for chave, valor in params.items():
            setattr(self, chave, valor)
        
        #criar uma pasta com número maior sem perder a ordenação
        self.directory = criar_pasta(self.directory)

        self._hyperparams = {
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'clip_range_vf': self.clip_range_vf,
            'normalize_advantage': self.normalize_advantage,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'use_sde': self.use_sde,
            'sde_sample_freq': self.sde_sample_freq,
            'rollout_buffer_class': self.rollout_buffer_class,
            'rollout_buffer_kwargs': self.rollout_buffer_kwargs,
            'target_kl': self.target_kl,
            'stats_window_size': self.stats_window_size,
            # 'tensorboard_log': self.tensorboard_log,
            'policy_kwargs': self.policy_kwargs,
            'verbose': self.verbose,
            'device': self.device
        }        

        self.train_env = Monitor(gymnasium.make(self.env_id))

        #* Reprodutibilidade
        torch.manual_seed(1)
        
        self.model = PPO_tunado(self.directory, 'MlpPolicy', self.train_env, self.reference_agent, self.calc_mutual_info, self._hyperparams)
    

    def plots(self):
        os.makedirs(f'{self.directory}/plots', exist_ok=True)
        data = pd.read_csv(f'{self.directory}/resultados.csv')
        
        # Função auxiliar local para renomear colunas para LaTeX para plots de pesos/gradientes
        def apply_latex_legend(data_frame: pd.DataFrame, component_type: str) -> pd.DataFrame:
            rename_map = {}
            for col in data_frame.columns:
                # Modificado: A regex agora é mais flexível para capturar a camada (layer_N ou layer) e o separador.
                # Captura: 'layer' + (opcionalmente '_N' ou ' N') + (opcionalmente '.' ou ' ') + (weight/bias)
                # Matches layer parameter columns like 'layer_0_weight', 'layer.1.bias', 'layer 2 weight', 'layer3bias', or 'layer_weight'.
                match = re.search(r'layer(?:[._\s](\d+))?(weight|bias)', col)
                if match:
                    # group(1) é o número da camada (N), se existir. Será None para a camada de saída.
                    layer_id = match.group(1) 
                    param_type = match.group(2)
                    
                    # Se não houver número de camada, assume-se que é a camada de saída (Out).
                    layer_label = layer_id if layer_id is not None else '\\text{Out}'
                    
                    # Base: W_i/W_Out ou b_i/b_Out
                    latex_base = f'\\text{{W}}_{{{layer_label}}}' if param_type == 'weight' else f'\\text{{b}}_{{{layer_label}}}'
                    
                    if component_type == 'grad':
                        # Para gradientes, adicionamos notação de norma, média ou desvio padrão
                        # O cabeçalho mostra 'mean', 'std' e sem sufixo (que é a norma ou o valor bruto)
                        if 'norm' in col:
                            latex_label = f'$|\\nabla {latex_base}|$'
                        elif 'mean' in col:
                            latex_label = f'$\\text{{Média}} (\\nabla {latex_base})$'
                        elif 'std' in col:
                            latex_label = f'$\\sigma (\\nabla {latex_base})$'
                        else:
                            # Fallback simples (Assume que o gradiente sem sufixo é a norma)
                            latex_label = f'$|\\nabla {latex_base}|$'
                    else: # weight
                        # Assumimos que qualquer coluna de peso sendo plotada para evolução é a sua magnitude (norma),
                        # pois os dados brutos ou a média/std dos pesos não são tipicamente plotados assim.
                        latex_label = f'$|{latex_base}|$'
                        
                    rename_map[col] = latex_label
            
            return data_frame.rename(columns=rename_map)

        # 1. Plots de Loss (inalterado)
        loss_data = data.filter(like='loss')
        loss_data.plot()
        fechar_plot(self.directory, 'loss', axle_x='Época', axle_y='Loss')
        
        # 2. Plots de Recompensa (inalterado)
        data_to_plot = pd.read_csv(f'{self.directory}/rewards.csv')
        rewards = data_to_plot.T
        means = rewards.apply(np.mean)
        stds = rewards.apply(np.std)
        plt.plot(means, label='Média', color='blue')
        plt.fill_between(
            range(len(means)),
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2,
            color='blue',
            label='$\\pm 1$ Desvio Padrão'
        )
        fechar_plot(self.directory, 'reward', 'Iteração', 'Recompensa')
        
        # 3. Plots de Pesos e Gradientes (COM REGEX REFORÇADO)
        
        # --- CORREÇÃO DE REGEX ---
        
        # Suffix para colunas de peso (captura _weight, _layer, e opcionalmente _norm)
        # NOTA: O prefixo ^(actor|critic) foi removido
        # Matches columns like:
        #   actor_weight_layer_0_weight
        #   actor_weight_layer_1_bias_norm
        #   critic_weight_layer1_weight
        #   actor_weight_layer_2_bias
        # The pattern matches column names ending with:
        #   '_weight' + optional separator + 'layer' + optional separator + optional layer number +
        #   optional separator + 'weight' or 'bias' + optional '_norm' at the end.
        weight_regex_suffix = r'_weight[._\s]?layer(?:[._\s](\d+))?[._\s]?(weight|bias)(?:_norm)?$'
        
        # Suffix para colunas de gradiente (captura _grad, _layer, e opcionalmente _norm, _mean, _std)
        grad_regex_suffix = r'_grad[._\s]?layer(?:[._\s](\d+))?[._\s]?(weight|bias)(?:_norm|_mean|_std)?$'
        
        # Ator - Pesos
        actor_weight_data = data.filter(regex=r'^actor' + weight_regex_suffix)
        actor_weight_data = apply_latex_legend(actor_weight_data, 'weight')
        actor_weight_data.plot()
        fechar_plot(self.directory, 'actor_weight', axle_y='Magnitude dos Pesos')

        # Ator - Gradientes
        actor_grad_data = data.filter(regex=r'^actor' + grad_regex_suffix)
        actor_grad_data = apply_latex_legend(actor_grad_data, 'grad')
        actor_grad_data.plot()
        fechar_plot(self.directory, 'actor_grad', axle_y='Magnitude dos Gradientes')

        # Crítico - Pesos
        critic_weight_data = data.filter(regex=r'^critic' + weight_regex_suffix)
        critic_weight_data = apply_latex_legend(critic_weight_data, 'weight')
        critic_weight_data.plot()
        fechar_plot(self.directory, 'critic_weight', axle_y='Magnitude dos Pesos')

        # Crítico - Gradientes
        critic_grad_data = data.filter(regex=r'^critic' + grad_regex_suffix)
        critic_grad_data = apply_latex_legend(critic_grad_data, 'grad')
        critic_grad_data.plot()
        fechar_plot(self.directory, 'critic_grad', axle_y='Magnitude dos Gradientes')

        # 4. Plots de Informação Mútua (MI) - Inalterado, já usa LaTeX
        
        # Filtrar apenas as colunas de MI do Ator. O regex agora busca o novo padrão: I_qualquer_coisa
        actor_mi_cols = data.filter(regex=r'^actor_I_.+').columns.tolist()
        # Gera combinações usando as chaves seguras (ex: I_X_h1, I_h1_h2)
        combinacoes_sequenciais_seguras = gera_combinacoes(colunas_mi_seguras=actor_mi_cols) 

        # Definir a coluna de cor baseada no índice
        size = len(data)
        color_col = np.arange(0, size)

        for prefix in ['actor', 'critic']:
            mi_data = data.filter(regex=rf'^{prefix}_I_.+')
            
            # Percorre as combinações sequenciais (as chaves SEGURAS sem o prefixo)
            for col1_safe, col2_safe in combinacoes_sequenciais_seguras:
                # Recompõe os nomes completos das colunas no DataFrame
                col1_full_name = f'{prefix}_{col1_safe}'
                col2_full_name = f'{prefix}_{col2_safe}'
                
                # Verifica se as colunas existem no DataFrame (caso o agente de ref não tenha 'Y_ref')
                if col1_full_name not in mi_data.columns or col2_full_name not in mi_data.columns:
                     continue
                     
                val1 = mi_data[col1_full_name]
                val2 = mi_data[col2_full_name]
                    
                plt.figure()
                plt.scatter(val1, val2, c=color_col, cmap='magma')
                
                # Cria a chave segura combinada para o nome do arquivo
                combined_safe_key = combinar_strings((col1_safe, col2_safe))
                
                # Geração da Label do Plot (Formato LaTeX)
                # Exemplo: I_X_h1 -> I(X,h_1)
                
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
                fechar_plot(self.directory, f'{prefix}_{combined_safe_key}', col_x_label_latex, col_y_label_latex)


        # 5. Colorbar (inalterado, apenas garantindo que o `size` esteja correto)
        fig, ax = plt.subplots(figsize=(12,1.75))
        norm = Normalize(vmin=0, vmax=size)
        cmap = plt.get_cmap('magma')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
        cbar.set_label('Épocas')
        plt.savefig(f'{self.directory}/plots/colorbar.pdf')
        plt.close()
        
    def treinamento(self):
        for seed in tqdm(self.seeds, desc="Training with different seeds"):
            self.model.set_random_seed(seed)
            self.model.learn(total_timesteps= self.timesteps, progress_bar= False)
        
        # recompensa
        df = pd.DataFrame(self.model.rewards_list)
        rewards_directory = os.path.join(self.directory, 'rewards.csv')
        df.to_csv(rewards_directory, mode= 'w', index=False, header= True)    

        self.model.save(os.path.join(self.directory, 'agente_treinado'))

        params = {chave: valor for chave, valor in self.__dict__.items() if not chave in ['train_env', 'model', '_hyperparams']}
        json_string = json.dumps(params, indent= 4)
        json_path = os.path.join(self.directory, f'{self.env_id}-{self.timesteps}.json')
        with open(json_path, 'w') as arquivo:      
            arquivo.write(json_string)

        self.plots()
        self.train_env.close()