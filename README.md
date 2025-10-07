# Códigos da Dissertação

## Instruções

Abandonai toda a esperança vós que aqui entrais

Estimação das métricas de um agente PPO em um ambiente da biblioteca Gymnasium.
As métricas calculadas são:
- informação mútua (IM) entre as camadas do ator;
- pesos e gradientes do ator;
- pesos e gradientes do crítico.

adicionarei um guia rápido de como instalar o repo via cmd depois que pegar meu diploma

## Manifesto de arquivos

1. `data`
existe para armazenar o resultado das simulações.Subdividida em:
    - `results`
    que apresenta a execução dos experimentos propriamente ditos;
    - `tensorboard_logs`
    que apresenta os arquivos para link com o tensorboard.
    
2. `src` 
apresenta os códigos eleaborados. Elementos presentes:
    - `quali` 
    pasta que contém os dados apresentados na Qualificação;
    - `class_experiment.py`
    que gerencia o agente adaptado, treinamento e plots;
    - `class_LQR_controller.py`
    que implementa via LQR um controle ótimo para o ambiente CartPole;
    - `class_ppo.py`
    que apresenta a técnica PPO modificada para medições de IM;
    - `CodeBase.ipynb`
    sendo o painel de controle principal dos experimentos;
    - `Geradado.ipynb`
    que faria o comparativo de múltiplas _seeds_;
    - `style.mplstyle`
    sendo o conjunto de configurações dos plots;
    - `teste.ipynb`
    que tem de tudo;
    - `utils.py`
    que tem de nada;

3. `.gitignore` 
autoexplicativo

4. `License.md` 
Licensa aplicada neste repositório

5. `pyproject.toml`
Configurações para fazer essa bagaça rodar

6. `README.md`
Este documento

7. `uv.lock`
existe

## 



# Melhorias previstas

- [ ] Melhora da legibilidade
- [ ] Coleta da IM das camadas do crítico
- [ ] Adicionar um marcador de convergência (vou adicionar uma métrica no estilo 5% da amplitude total)
- [ ] Gravar o treinamento
- [ ] Implementação da paralelização (o esquema de `n_envs` não tá funcionando)
- [ ] Verificar alternativas para inicialização da rede. Atualmente cada `enviroment` gera uma única rede independente da `seed` do ambiente. Bom para reprodução, mas pode ser um problema sobre a generalização.
- [ ] Ajustar atualização e registro dos modelos. Usar a classe `PPO_tunado` acabou sobrescrevendo a opção de atualização de pesos do próprio modelo. Problema bônus, ao resolver isso verificar se essa ideia de usar múltiplas seeds em uma única inicialização (`np.arange(0,100)`) não vai ficar contaminada pela run anterior

### Problemas de visualização
- [ ] Alternativas para melhorar a visibilidade da evolução da IM (rollout maior? Menos épocas?)
- [ ] Ideias sobre como mostrar a evolução da IM em múltiplas seeds (são 4 elementos variando, se fizer um vídeo fica viável)
- [ ] Verificar se os plots com desvio padrão (recompensa) estão sendo medidos de maneira correta


**Citação**

Vai entrar um bib aqui, eventualmente.