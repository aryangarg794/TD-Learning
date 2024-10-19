import gymnasium as gym
import numpy as np

from abc import abstractmethod
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
        max_epsilon: float = 0.2
        ) -> None:
        
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
            action = int(np.argmax(self.q_function[state]))
        return action 
    
    def decay_epsilon(self: Self) -> None:
        self.epsilon = max(self.max_epsilon, self.epsilon - self.epsilon_decay)
        
    def predict(
        self: Self,
        state: Any
        ) -> np.int64 | int:
        return int(np.argmax(self.q_function[state]))

class QLearningAgent(TemporalDifferenceAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
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
    
    def update(
        self: Self,
        state: Any,
        action: Any,
        reward: SupportsFloat | int, 
        terminated: bool,
        next_state: Any
    ) -> None:
        pass