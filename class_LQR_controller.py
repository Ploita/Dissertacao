# code from https://github.com/JoKoum/reinforcement-learning-vs-control-theory/blob/master/cartpole_control_theory.py
import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np
import gymnasium

class Controller:
    '''Implements a feedback controller'''
    def __init__(self, environment):
        # gravity
        self.g = 10
        # pole length
        self.lp = environment.env.get_wrapper_attr('length')
        # pole mass
        self.mp = environment.env.get_wrapper_attr('masspole')
        # cart mass
        self.mk = environment.env.get_wrapper_attr('masscart')
        # total mass
        self.mt = environment.env.get_wrapper_attr('total_mass')
        
    def state_controller(self):
        # state matrix
        a = self.g/(self.lp*(4.0/3 - self.mp/(self.mp+self.mk)))
        A = np.array([[0, 1, 0, 0],
            [0, 0, a, 0],
            [0, 0, 0, 1],
            [0, 0, a, 0]])
            
        # input matrix
        b = -1/(self.lp*(4.0/3 - self.mp/(self.mp+self.mk)))
        B = np.array([[0], [1/self.mt], [0], [b]])
        
        # choose R (weight for input)
        R = np.eye(1, dtype=int)
        # choose Q (weight for state)
        Q = 5*np.eye(4, dtype=int)
        
        # solve ricatti equation
        P = linalg.solve_continuous_are(A, B, Q, R)     #! Tem algo aqui que crasha o kernel
        
        # calculate optimal controller gain
        K = np.dot(np.linalg.inv(R), np.dot(B.T, P))

        return K
        
    def apply_state_controller(self, x):
        # Adaptei pro modo torch
        K = self.state_controller()
        
        # Calcular a força para todos os estados em um único passo
        forces = -np.dot(K, x)
        
        # Determinar a ação (0 ou 1) com base na força
        actions = int(forces[0] > 0)

        return actions

def run_experiment(rounds = 1000):
    '''Perform an experiment. Control the cart-pole system'''
    # get environment
    env = gymnasium.make('CartPole-v1', render_mode = 'human')
    obs = env.reset(seed= 0)[0]
    controller = Controller(env)

    position_list = []
    velocity_list = []
    angle_list = []
    angular_velocity_list = []
    steps = []
    flag = True
    
    for i in range(rounds):        
        # get force direction (action) and force value (force)
        action = controller.apply_state_controller(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # store state, time
        position_list.append(obs[0])
        velocity_list.append(obs[1])
        angle_list.append(obs[2])
        angular_velocity_list.append(obs[3])
        steps.append(i)
        
        done = terminated or truncated

        if done and flag:
            print(f'Threshold reached after {i+1} iterations.')
            flag = False
        if i > 498:
            break
        
    env.close()
    
    fig, ax = plt.subplots(2,2, figsize=(15,8))
    ax[0][0].plot(steps, position_list)
    ax[0][0].set_xlabel('Time steps')
    ax[0][0].set_ylabel('Position (m)')
    ax[0][0].grid()
    
    ax[0][1].plot(steps, velocity_list, 'r')
    ax[0][1].set_xlabel('Time steps')
    ax[0][1].set_ylabel('Velocity (m/s)')
    ax[0][1].grid()
    
    ax[1][0].plot(steps, angle_list, 'g')
    ax[1][0].set_xlabel('Time steps')
    ax[1][0].set_ylabel('Angle (rad)')
    ax[1][0].grid()
    
    ax[1][1].plot(steps, angular_velocity_list, 'y')
    ax[1][1].set_xlabel('Time steps')
    ax[1][1].set_ylabel('Angular Velocity (rad/s)')
    ax[1][1].grid()
    
    plt.suptitle('Observations per step')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__': 
    run_experiment() 