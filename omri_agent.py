from pommerman.agents import BaseAgent
import numpy as np
import random
import pommerman.utility as util
import os.path

def flip_coin(p):
    r = random.random()
    return r > p

filename = 'qvalues.npy'

class NewAgent(BaseAgent):
    

    #TODO:
    # - check flames instead of bombs
    # -  make sure shown state is cur_state

    def __init__(self, *args, **kwargs):
        super(NewAgent, self).__init__(*args, **kwargs)
        self.last_action = 0
        self.new_action = 0
        self.cur_state = (0,)
        self.last_state = (0,)
        self.epsilon = 0.8
        self.discount = 1
        self.alpha = 1
        self.done = False
        self.q_values = dict()
        if os.path.isfile(filename):
            self.q_values = np.load(filename).item()
            
            
    def save_qvalues(self):
        np.save(filename, self.q_values)

    def extract_state(self, obs):
        dirs = [1,2,3,4]
        board = obs["board"]
        pos = np.array(obs["position"])
        bomb_life = obs["bomb_life"]
        bomb_strength = obs["bomb_blast_strength"]
        valid_directions = [util.is_valid_direction(board, pos, d) for d in dirs]
        dangers = np.ones(4) * 50
        for i,d in enumerate(np.array([[0,1],[0,-1],[1,0],[-1,0]])):
            temp_pos = pos + d
            bomb_range = 0
            
            while util.position_on_board(board, temp_pos):
                bomb_range += 1
                if bomb_strength[temp_pos[0],temp_pos[1]] >= bomb_range:
                    dangers[i] = min(dangers[i], bomb_life[temp_pos[0],temp_pos[1]])
                temp_pos += d
        state = (tuple(valid_directions), tuple(dangers))
        return state
            
    def set_start_state(self, obs):
        self.cur_state = self.extract_state(obs)

    def update_q_value(self, reward):
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
            self.episode_end();
        

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

    def episode_end(self):
        self.done = True
        self.last_action = 0
        self.new_action = 0
        self.cur_state = (0,)
        self.last_state = (0,)

