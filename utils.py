from stable_baselines3.common.callbacks import BaseCallback
from pushbullet import Pushbullet
import pandas as pd
import gymnasium
import torch
import os

# usar com o decorador @wrap_alerta
def wrap_alerta(func):
    """função para alertar via pushbullet
    """
    def msg(key: int, f):
        with open('chave.txt', 'r') as arquivo:
            chave = arquivo.read()
            pb = Pushbullet(chave)
            title = 'Fim da execução!' if key == 1 else 'Deu ruim!'
            pb.push_note(title, f)        

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg(0, e)
        finally:
            msg(1,'cabou')
    return wrapper

def alerta():
    with open('chave.txt', 'r') as arquivo:
        chave = arquivo.read()
        pb = Pushbullet(chave)
        title = 'Fim da execução!'
        pb.push_note(title, 'cabou') 


def fib(n: int):
    """Gera os n primeiros números da sequência de Fibonacci utilizando um loop.

    Args:
        n: Número de termos da sequência.

    Returns:
        Uma lista com os n primeiros números da sequência.
    """
    assert n > 0, 'N precisa ser maior do que 0'
    if n == 1:
        return [1]
    fib = [1, 2]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

def tensor_to_numpy(val:torch.Tensor):
    return val.detach().clone().cpu().numpy()

class CustomCallback(BaseCallback):
    def __init__(self, coleta: bool, env_id: str, directory: str, verbose: int = 0):
        super().__init__(verbose)
        self.counter = 0
        self.coleta = coleta
        self.env_id = env_id
        self.directory = directory

    def _on_step(self) -> bool:
        return super()._on_step()
    
    def _on_rollout_start(self) -> None:
        if not self.coleta:
            return super()._on_rollout_end()
        env = gymnasium.make(self.env_id)
        self.model.set_random_seed(0)   #* Reprodutibilidade 
        seeds = fib(100)                #* Reprodutibilidade 
        if not os.path.exists(f'{self.directory}/Coleta'):
            os.makedirs(f'{self.directory}/Coleta')

        dir = f'{self.directory}/Coleta/coleta_treino_{str(self.counter).zfill(3)}.csv'
        for ite, seed in enumerate(seeds):
            obs = env.reset(seed=seed)[0]   #* Reprodutibilidade 
            done = False
            i = 1
            while not done:
                tensor_obs = self.model.policy.obs_to_tensor(obs)[0]
                action1 = self.model.policy.predict_values(tensor_obs) #type:ignore
                action2 = self.model.policy.get_distribution(tensor_obs) #type:ignore
                
                action, _ = self.model.predict(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                df = pd.DataFrame({
                        'Iteração': ite,
                        'Passo': i,
                        'Posição': obs[0],
                        'Velocidade': obs[1],
                        'Ângulo': obs[2],
                        'Velocidade Angular': obs[3],
                        'Distribuição': action2.distribution.probs[0,0].detach().clone().cpu().numpy(), #type: ignore
                        'Entropia': action2.distribution.entropy().detach().clone().cpu().numpy() #type: ignore
                                }, index=[0])
                df.to_csv(dir, mode= 'a' if os.path.exists(dir) else 'w', index=False, header= not os.path.exists(dir))
                done = terminated or truncated
                i += 1
            env.close()
        self.counter += 1
