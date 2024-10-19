import os
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from q_learning.td_learning import QLearningAgent, SARSAAgent

agents = {
    'qlearn' : 'Q-Learning',
    'sarsa' : 'SARSA',
    'esarsa' : 'Expected SARSA'
}

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', type=str, default='qlearn', 
                    help='select agent: qlearn -> Q-Learning, sarsa -> SARSA, esarsa -> Expected SARSA')

parser.add_argument('-e', '--numeps', type=int, default=100, help='number of episodes')
parser.add_argument('-l', '--lr', type=float, default=0.9, help='learning rate')
parser.add_argument('-eps', '--epsilon', type=float, default=0.65, help='starting epsilon')
parser.add_argument('-d', '--discount', type=float, default=0.95, help='discount factor')
parser.add_argument('-m', '--maxepsilon', type=float, default=0.2, help='maximum epsilon for epsilon decay')
parser.add_argument('-g', '--game', type=str, default='CliffWalking-v0', help='select game to run on')
parser.add_argument('-r', '--render', type=str, default='human', help='Render mode')
args = parser.parse_args()
args_dict = vars(args)

agent_type = args.agent
num_eps = args.numeps
game = args.game
render = args.render

env = gym.make(game, render_mode=render)
env = gym.wrappers.RecordEpisodeStatistics(env=env, buffer_length=num_eps)

if agent_type == 'qlearn':
    agent = QLearningAgent(environ=env, n_episodes=num_eps, 
                           epsilon=args.epsilon, 
                           max_epsilon=args.maxepsilon, 
                           discount_factor=args.discount,
                           learning_rate=args.lr
                           )
elif agent_type == 'sarsa':
    agent = SARSAAgent(environ=env, n_episodes=num_eps,
                       epsilon=args.epsilon, 
                       max_epsilon=args.maxepsilon, 
                       discount_factor=args.discount,
                       learning_rate=args.lr) #type:ignore
    
elif agent_type == 'esarsa':
    raise RuntimeError('Expected SARSA not implemented yet')


if __name__ == '__main__':
    text = f'Training with Agent: {agents[agent_type]}'
    terminal_width = os.get_terminal_size().columns
    padding = (terminal_width - len(text)) // 2
    print('=' * terminal_width)
    print(' ' * padding + text)
    print('=' * terminal_width + '\n')
    
    for setting, value in args_dict.items():
        setting_text = f'{setting}: {value}'
        padding_item = (terminal_width - len(setting_text)) // 2
        print(' ' * padding_item + setting_text) 
    
    for episode in tqdm(range(num_eps)):
        curr_state, info = env.reset()
        done = False
        
        print(f'Episode: {episode+1} | epsilon: {agent.epsilon}')
        
        while not done:
            action = agent.epsilon_greedy(curr_state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            agent.update(curr_state, action, reward, terminated, next_state)
            
            done = terminated or truncated
            curr_state = next_state
        
        agent.decay_epsilon()