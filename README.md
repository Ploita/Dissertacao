# Códigos da Dissertação

## Instruções

Abandonai toda a esperança vós que aqui entrais

Estimação das métricas de um agente PPO em um ambiente da biblioteca Gymnasium.
As métricas calculadas são:
- informação mútua (IM) entre as camadas do ator;
- pesos e gradientes do ator;
- pesos e gradientes do crítico.

## Manifesto de arquivos

1. `data`
existe para armazenar o resultado das simulações.
    
2. `src` 
apresenta os códigos elaborados. 

3. `notebook` 
apresenta os códigos preliminares. 

4. `.gitignore` 
autoexplicativo

5. `LICENSE.md` 
Licensa aplicada neste repositório

6. `pyproject.toml`
Configurações para fazer essa bagaça rodar

## 



# To-do list

- [ ] Melhora da legibilidade
- [x] Coleta da IM das camadas do crítico
- [ ] Adicionar um marcador de convergência (adicionar uma métrica no estilo 5% da amplitude total?)
- [ ] Gravar o treinamento
- [ ] Implementação da paralelização (o esquema de `n_envs` não tá funcionando)
- [X] Verificar alternativas para inicialização da rede. Atualmente cada `enviroment` gera uma única rede independente da `seed` do ambiente. Bom para reprodução, mas pode ser um problema sobre a generalização
- [ ] Ajustar atualização e registro dos modelos. Usar a classe `PPO_tunado` acabou sobrescrevendo a opção de atualização de pesos do próprio modelo. Problema bônus, ao resolver isso verificar se essa ideia de usar múltiplas seeds em uma única inicialização (`np.arange(0,100)`) não vai ficar contaminada pela run anterior
- [ ] Verificar se o `device` pode acelerar o treinamento
- [ ] Ajustar a questão do agente de referência para comparar o ator e crítico do agente em treinamento
- [ ] Docker

### Problemas de visualização
- [ ] Alternativas para melhorar a visibilidade da evolução da IM (rollout maior? Menos épocas?)
- [x] Ideias sobre como mostrar a evolução da IM em múltiplas seeds (são 4 elementos variando, se fizer um vídeo fica viável)
- [x] Verificar se os plots com desvio padrão (recompensa) estão sendo medidos de maneira correta
- [x] Melhorar a legenda para viés e pesos de cada camada (sair de `actor_weight_layer_0.weight` pra `w_0` ou coisa similar)


**Citação**

Você pode citar este código utilizando minha [dissertação](https://repositorio.unicamp.br/acervo/detalhe/1522646) como:

```
@thesis{fernandes2025,
	location = {Campinas, {SP}},
	title = {Análise do treinamento de redes neurais profundas no paradigma de aprendizado por reforço através do plano de informação},
	pagetotal = {93},
	institution = {Universidade Estadual de Campinas},
	type = {Dissertação (mestrado)},
	author = {Fernandes, Arthur Felipe dos Santos},
	date = {2025},
    url = {https://repositorio.unicamp.br/acervo/detalhe/1522646},
}
```