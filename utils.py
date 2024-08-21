import gymnasium
import keyboard
from pushbullet import Pushbullet
from stable_baselines3.ppo import PPO

def alerta():
    with open('chave.txt', 'r') as arquivo:
        chave = arquivo.read()
        pb = Pushbullet(chave)
        pb.push_note("Fim da execução!", "Seu código no computador do laboratório terminou de rodar.")
    print('Done.')


def verificar_modelo_ppo(env_name: str, best_model: PPO, hardcore: bool = False):
    """Gera a visualização do agente treinado no ambiente

    Parameters
    ----------
    best_model : PPO
        modelo
    """
    env = gymnasium.make(env_name, render_mode = 'rgb_array', hardcore = hardcore)
    obs = env.reset()[0]
    i = 0
    seeds = [13, 21, 34, 55, 89]
    while i < 5:
        best_model.set_random_seed(seeds[i])
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            i = i + 1
        if keyboard.is_pressed('esc'):
            break
    env.close()