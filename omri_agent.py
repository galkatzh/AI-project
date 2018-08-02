from pommerman.agents import BaseAgent
import numpy as np
import random
import pommerman.utility as util

def flip_coin(p):
    r = random.random()
    return r > p

class NewAgent(BaseAgent):

    #TODO:
    # - stop computing after losing (because game keep running)
    # - when is reward returned, how?
    # in train script, call update_q_values. not in act()
    # - store q values in file
    # - how to get start state?
    # - check flames instead of bombs

    def __init__(self, *args, **kwargs):
        super(NewAgent, self).__init__(*args, **kwargs)
        self.last_action = 0
        self.new_action = 0
        self.cur_state = (0,)
        self.last_state = (0,)
        self.q_values = dict()
        self.epsilon = 0.8
        self.discount = 1
        self.alpha = 1
        self.done = False

    def extract_state(self, obs):
#        def is_bomb_adjacent(board, pos):
#            x, y = pos
#            adj = [(x+1,y),(x-1,y),(x,y+1),(x,y-1),(x,y)]
#            danger = [0] * len(adj)
#            for i, (x,y) in enumerate(adj):
#                if x>=0 and x<=10 and y>=0 and y<=10:
#                    if board[x,y] == 1:
#                        danger[i] = 1
#            return danger
#        return is_bomb_adjacent(bomb_life, obs["position"])
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
            


    def update_q_value(self, reward):
#        if (0,0,1,0,0) in self.q_values:
#            if -1 in self.q_values[(0,0,1,0,0)]:
#                print("ASDASDASDASDASDASDASDASDASDASDASDASDASDa")
#            print (self.q_values[(0,0,1,0,0)])
#            print(reward)
        if self.done:
            return
        if tuple(self.cur_state) not in self.q_values:
            self.q_values[tuple(self.cur_state)] = [0] * 6
        if tuple(self.last_state) not in self.q_values:
            self.q_values[tuple(self.last_state)] = [0] * 6
        best_next_action = np.argmax(self.q_values[tuple(self.cur_state)])    
        td_target = reward + self.discount * self.q_values[tuple(self.cur_state)][best_next_action]
        td_delta = td_target - self.q_values[tuple(self.last_state)][self.last_action]
        self.q_values[tuple(self.last_state)][self.last_action] += self.alpha * td_delta
        if reward != 0:
            self.episode_end();
        

    def act(self, obs, action_space):
#        import IPython
#        IPython.embed()
#        print(obs['bomb_life'])
        self.done = False
        self.last_action = self.new_action
        self.last_state = self.cur_state
        self.cur_state = self.extract_state(obs)
        #self.update_q_value(self.last_state, self.last_action, cur_state)
        if self.extract_state(obs) in self.q_values and flip_coin(self.epsilon):
            self.new_action = np.argmax(self.q_values[tuple(self.extract_state(obs))])
        else:
            self.new_action = action_space.sample()
        return self.new_action

    def episode_end(self):
        self.done = True
        self.last_action = 0
        self.new_action = 0
        self.cur_state = (0,)
        self.last_state = (0,)

