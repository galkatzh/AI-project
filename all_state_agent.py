from pommerman.agents import BaseAgent
import numpy as np
import random
import pommerman.utility as util
import pommerman.constants as consts
import os.path


#TODO:
    # -  make sure shown state is cur_state
    # - where powerup
    # - check if enemies can escape
    # - use reward for killing other players
    
filename = 'full_qvalues.npz'
dirs = [1,2,3,4]  #up, down, left, right    

def flip_coin(p):
    r = random.random()
    return r > p

def freeze(d):
    if isinstance(d, dict):
        return frozenset((key, freeze(value)) for key, value in d.items())
    elif isinstance(d, np.ndarray):
        return tuple(d.flatten())
    elif isinstance(d, list):
        return tuple(freeze(value) for value in d)
    return d

class NewAgent(BaseAgent):

    def __init__(self, discount, epsilon, alpha):
        super(NewAgent, self).__init__()
        self.last_action = 0
        self.new_action = 0
        self.cur_state = (0,)
        self.last_state = (0,)
#        self.epsilon = 0.8
#        self.discount = 1
#        self.alpha = 1
        self.epsilon = epsilon
        self.discount = discount
        self.alpha = alpha
        self.done = False
        self.q_values = dict()
        if os.path.isfile(filename):
            self.q_values = np.load(filename)['q'].item()
            
    def save_qvalues(self):
        np.savez_compressed(filename, q=self.q_values)

    def extract_state(self, obs):
#        dirs = [1,2,3,4]  #up, down, left, right
        return freeze(obs)
            
    def set_start_state(self, obs):
        self.cur_state = self.extract_state(obs)

    def update_q_value(self, reward, new_state = None, old_state = None, last_action=None):
        if new_state == None or old_state == None or last_action==None:
            if self.done:
                return
            if self.cur_state not in self.q_values:
                self.q_values[self.cur_state] = [0] * 6
            if self.last_state not in self.q_values:
                self.q_values[self.last_state] = [0] * 6
            best_next_action = np.argmax(self.q_values[self.cur_state])    
            td_target = reward + self.discount * self.q_values[self.cur_state][best_next_action]
            td_delta = td_target - self.q_values[self.last_state][self.last_action]
            self.q_values[self.last_state][self.last_action] += self.alpha * td_delta
            if reward != 0:
                self.episode_end(reward);
        else:
            if new_state not in self.q_values:
                self.q_values[new_state] = [0] * 6
            if old_state not in self.q_values:
                self.q_values[old_state] = [0] * 6
            best_next_action = np.argmax(self.q_values[new_state])    
            td_target = reward + self.discount * self.q_values[new_state][best_next_action]
            td_delta = td_target - self.q_values[old_state][last_action]
            self.q_values[old_state][last_action] += self.alpha * td_delta
        

    def act(self, obs, action_space):
#        import IPython
#        IPython.embed()
        self.done = False
        self.last_action = self.new_action
        self.last_state = self.cur_state
        self.cur_state = self.extract_state(obs)
        if self.extract_state(obs) in self.q_values and flip_coin(self.epsilon):
            self.new_action = np.argmax(self.q_values[self.extract_state(obs)])
        else:
            self.new_action = action_space.sample()
        return self.new_action

    def episode_end(self, reward):
        if not self.done:
            self.done = True
            self.last_action = 0
            self.new_action = 0
            self.cur_state = (0,)
            self.last_state = (0,)
