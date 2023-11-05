import random
import torch.nn as nn
import numpy as np
import torch
import pickle
import os

from collections import deque
from torch.distributions.normal import Normal

actions_map = [
0,     #'noop'
32,   #'down'
16,   #'up'
1,    #'jump'
3,    #'spin'
64,   #'left'
65,   #'jumpleft'
66,   #'runleft'
67,   #'runjumpleft'
128,  #'right'
129,  #'jumpright'
130,  #'runright'
131,  #'runjumpright'
384,  #'spinright'
386,  #'runspinright'
320,  #'spinleft'
322   #'spinrunleft'
]
actions_dict = {actions_map[i]: i  for i in range(len(actions_map))}

class DQNNetwork(nn.Module):
    """Network for the DQN agent."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__() #type:ignore

        hidden_space1 = 128
        hidden_space2 = 256

        self.network = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, action_space_dims),
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state.float())

    def get_network_weights(self) -> 'list[torch.Tensor]':
        return [param.data.clone().detach() for param in self.network.parameters()]

class DQNAgent:
    """Agent that learns to solve the environment using DQN."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        self.memory = deque(maxlen=1000000)  # experience replay #type: ignore
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # minimal exploration rate
        self.epsilon_decay = 1 - 1/100  # exploration decay
        self.batch_size = 10  # batch size for the experience replay
        self.update_freq = 10  # frequency of updating the target network
        self.tau = 0.15 # update rate of the target network

        # ? Seria interessante ter uma contagem das épocas treinadas para que o agente
        # ? decida zerar o epsilon mínimo?

        # networks
        self.action_space_dims = action_space_dims
        self.q_network = DQNNetwork(obs_space_dims, action_space_dims)
        self.target_network = DQNNetwork(obs_space_dims, action_space_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    def store_transition(self, state: 'list[float]', action: float, reward: float, next_state: 'list[float]', done: bool):
        self.memory.append((state, action, reward, next_state, done)) #type: ignore

    def choose_action(self, state: 'list[float]') -> int:
        if np.random.rand() <= self.epsilon:
            return actions_map[random.randint(0,self.action_space_dims - 1)]
        else:
            state = torch.tensor(np.array([state]))  # type: ignore
            q_values = self.q_network(state).detach().numpy()
            return actions_map[int(np.argmax(q_values))]
    
    def learn(self):
        # Make sure the replay buffer is at least batch size large
        if len(self.memory) < self.batch_size: #type:ignore
            return
        minibatch = random.sample(self.memory, self.batch_size) #type:ignore
        for state, action, reward, next_state, done in minibatch: #type:ignore
            state = torch.tensor(np.array(state))  # type: ignore
            next_state = torch.tensor(np.array(next_state))  # type: ignore
            target = reward
            if not done:
                target = (
                    reward
                    + self.gamma * torch.max(self.target_network(next_state))
                )
            current = self.q_network(state)[actions_dict[action]]
            target = torch.tensor(
                target, dtype=torch.float32
            )  # convert target to tensor
            loss = torch.nn.functional.mse_loss(current, target)
         
            self.optimizer.zero_grad()
            loss.backward() #type:ignore
            self.optimizer.step()
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def update_target_network(self):       
        target_net_state_dict = self.target_network.state_dict()
        q_net_state_dict = self.q_network.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = q_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_network.load_state_dict(target_net_state_dict)
    
    def save(self):
        with open('dqn_agent.pkl', 'wb') as file:
            pickle.dump(self, file)

    def load(self):
        if os.path.exists('dqn_agent.pkl'):
            with open('dqn_agent.pkl', 'rb') as file:
                return pickle.load(file)
        else:
            return DQNAgent(169, 17)