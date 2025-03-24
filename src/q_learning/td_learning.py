import gymnasium as gym
import os
import numpy as np

from abc import abstractmethod
from datetime import datetime
from typing import Any, Self, SupportsFloat

class TemporalDifferenceAgent:
    def __init__(
        self: Self,
        environ: gym.Env, 
        n_episodes: int = 100,
        epsilon: float = 0.65 ,
        learning_rate: float = 0.9,
        discount_factor: float = 0.95,
        random: bool = False,
        max_epsilon: float = 0.01
        ) -> None:
        
        self._agent_name = 'Agent'
        self.env = environ
        self.nr_of_actions = environ.action_space.n #type:ignore
        self.nr_of_states = environ.observation_space.n #type:ignore
        
        self.epsilon = epsilon
        self.epsilon_decay = self.epsilon / (n_episodes / 2)
        self.max_epsilon = max_epsilon
        self.lr = learning_rate
        self.gamma = discount_factor
        
        # in this case [0, 1] indicates state 0 with action 1
        if random:
            self.q_function = np.random.rand(self.nr_of_states, self.nr_of_actions)
        else:
            self.q_function = np.zeros((self.nr_of_states, self.nr_of_actions))
    
    @abstractmethod
    def update(
        self: Self,
        state: Any, #techinally this is ObsType from gym but I couldn't find the the type
        action: Any,
        reward: SupportsFloat | int, 
        terminated: bool,
        next_state: Any
    ) -> None:
        
        pass
    
    def epsilon_greedy(
        self: Self,
        state: Any
    ) -> np.int64:
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            arr = self.q_function[state]
            actions = np.where(arr == arr.max())[0]
            action = np.random.choice(actions)
        return action 
    
    def decay_epsilon(self: Self) -> None:
        self.epsilon = max(self.max_epsilon, self.epsilon - self.epsilon_decay)
        
    def predict(
        self: Self,
        state: Any
        ) -> np.int64 | int:
        return int(np.argmax(self.q_function[state]))
    
    def save(
        self: Self
    ) -> None:
        
        folder_name = 'agents'
        
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(folder_name, f'{self._agent_name}_{timestamp}.txt')
        np.savetxt(filename, self.q_function, fmt='%d')
        # print(f"Agent saved as {self._agent_name}_{timestamp}.txt")
        
    def load(
        self: Self, 
        fname: str
    ) -> None:
        self.q_function = np.loadtxt(fname)
        

class QLearningAgent(TemporalDifferenceAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self._agent_name = 'Qlearn'
        
    def update(
        self: Self,
        state: Any,
        action: Any,
        reward: SupportsFloat | int, 
        terminated: bool,
        next_state: Any
    ) -> None:
        greedy_action = np.max(self.q_function[next_state]) if not terminated else 0
        self.q_function[state, action] += self.lr * (reward + self.gamma * greedy_action - self.q_function[state, action])
        
class SARSAAgent(TemporalDifferenceAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self._agent_name = 'SARSA'
        
    def update(
        self: Self,
        state: Any,
        action: Any,
        reward: SupportsFloat | int, 
        terminated: bool,
        next_state: Any
    ) -> None:
        
        # for sarsa we need to get the next state from current policy which is e-greedy
        policy_action = self.epsilon_greedy(next_state)
        policy_reward = self.q_function[next_state, policy_action] if not terminated else 0
        self.q_function[state, action] += self.lr * (reward + self.gamma * policy_reward - self.q_function[state, action])

class ExpectedSARSAAgent(TemporalDifferenceAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)    
        
        self._agent_name = 'ESARSA'
    
    def _epsilon_greedy(
        self,
        action: Any
    ) -> None:
        if np.random.random() < self.epsilon:
            transition_model = 1/self.nr_of_actions * np.ones(self.nr_of_actions)
        else:
            transition_model = np.zeros(self.nr_of_actions)
            transition_model[action] = 1.0
        return transition_model 
    
    def update(
        self: Self,
        state: Any,
        action: Any,
        reward: SupportsFloat | int, 
        terminated: bool,
        next_state: Any
    ) -> None:
        policy_action = self.epsilon_greedy(next_state)
        if not terminated:
            transition_model = self._epsilon_greedy(policy_action)
        else:
            transition_model = np.zeros(self.nr_of_actions)
        
        policy_reward = sum([transition_model[i] * self.q_function[next_state, i] for i in range(self.nr_of_actions)]) #type:ignore
        self.q_function[state, action] += self.lr * (reward + self.gamma * policy_reward - self.q_function[state, action])

class DoubleQAgent(TemporalDifferenceAgent):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self._agent_name = 'DoubleQ'
        self.q1 = QLearningAgent(**kwargs)
        self.q2 = QLearningAgent(**kwargs)
        
        self.n_sa = np.zeros((2, self.nr_of_states, self.nr_of_actions))
        
    def update(
        self: Self, 
        state: Any,
        action: Any,
        reward: SupportsFloat | int, 
        terminated: bool,
        next_state: Any
    ) -> None:
        func = np.random.choice([0, 1])
        # self.update_lr(state, action, func)
        
        if func == 0:
            greedy_action = np.argmax(self.q1.q_function[next_state]) if not terminated else 0
            self.q1.q_function[state, action] += self.lr * (reward + self.gamma * self.q2.q_function[next_state, greedy_action] \
                - self.q1.q_function[state, action])
        else: 
            greedy_action = np.argmax(self.q2.q_function[next_state]) if not terminated else 0
            self.q2.q_function[state, action] += self.lr * (reward + self.gamma * self.q1.q_function[next_state, greedy_action] \
                - self.q2.q_function[state, action])
            
        self.q_function = (self.q1.q_function + self.q2.q_function) / 2
    
    def update_lr(self, state: Any, action: Any, func: int) -> None:
        self.lr = 1/(self.n_sa[func, state, action])**0.8 if self.n_sa[func, state, action] > 0 else self.lr
        self.n_sa[func, state, action] += 1
        
