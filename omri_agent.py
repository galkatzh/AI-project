from pommerman.agents import BaseAgent
import numpy as np
import random
import pommerman.utility as util
import pommerman.constants as consts
import os.path


#TODO:
    # -  make sure shown state is cur_state
    # - check flames instead of bombs
    # - check enemy or wood wall in range
    # - in functions above, stop if reach rigid
    # - where powerup
    
filename = 'qvalues.npy'
dirs = [1,2,3,4]  #up, down, left, right    

def flip_coin(p):
    r = random.random()
    return r > p

def get_valid_directions(board, pos):
    return tuple([util.is_valid_direction(board, pos, d) for d in dirs])

def get_bombs(board,pos, all_bombs_strength, bomb_life):
    dangers = np.ones(4) * 50
    for i,d in enumerate(np.array([[0,1],[0,-1],[1,0],[-1,0]])):
        temp_pos = pos + d
        bomb_range = 0
        while util.position_on_board(board, tuple(temp_pos)) and not util.position_is_fog(board, tuple(temp_pos)) and not util.position_is_rigid(board, tuple(temp_pos)):
            bomb_range += 1
            if all_bombs_strength[tuple(temp_pos)] >= bomb_range:
                dangers[i] = min(dangers[i], bomb_life[temp_pos[0],temp_pos[1]])
            temp_pos += d
    return tuple(dangers)

def get_flames(board,pos):
    dangers = [False] * 4
    for i, d in enumerate(dirs):
        new_pos = tuple(util.get_next_position(pos, consts.Action(d)))
        if util.position_on_board(board, new_pos) and util.position_is_flames(board, new_pos):
            dangers[i] = True
    return tuple(dangers)

def get_quarter(pos):
    r,c = pos
    quarters = [[1,2],[3,4]]
    return quarters[r>=5][c>=5]

def is_enemy_in_range(board, pos, blast_radius, enemies):
    for d in np.array([[0,1],[0,-1],[1,0],[-1,0]]):
        r = blast_radius
        temp_pos = pos + d
        while r > 0 and util.position_on_board(board, tuple(temp_pos)) and not util.position_is_rigid(board, tuple(temp_pos)):
            if util.position_is_enemy(board, tuple(temp_pos), enemies):
                return True
            temp_pos = pos + d
            r -= 1
#        for r in range(1,blast_radius+1):
#            temp_pos = tuple(pos + r*d)
#            if :
#                return True
    return False

def is_wood_in_range(board, pos, blast_radius):
    for d in np.array([[0,1],[0,-1],[1,0],[-1,0]]):
        r = blast_radius
        temp_pos = pos + d
        while r > 0 and util.position_on_board(board, tuple(temp_pos)) and not util.position_is_rigid(board, tuple(temp_pos)):
            if util.position_is_wood(board, tuple(temp_pos)):
                return True
            temp_pos = pos + d
            r -= 1
    return False
    
def is_thing_in_range(board, pos, blast_radius, enemies):
    for d in np.array([[0,1],[0,-1],[1,0],[-1,0]]):
        r = blast_radius
        temp_pos = pos + d
        while r > 0 and util.position_on_board(board, tuple(temp_pos)) and not util.position_is_rigid(board, tuple(temp_pos)):
            if util.position_is_enemy(board, tuple(temp_pos), enemies) or util.position_is_wood(board, tuple(temp_pos)):
                return True
            temp_pos = pos + d
            r -= 1
    return False


class NewAgent(BaseAgent):

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
#        dirs = [1,2,3,4]  #up, down, left, right
        board = obs["board"]
        pos = np.array(obs["position"])
        bomb_life = obs["bomb_life"]
        all_bombs_strength = obs["bomb_blast_strength"]
        can_kick = obs['can_kick']
        blast_radius = obs['blast_strength']
        ammo = obs['ammo']
        enemies = obs['enemies']
#        valid_directions = [util.is_valid_direction(board, pos, d) for d in dirs]
        
        valid_directions = get_valid_directions(board, pos)
        dangerous_bombs = get_bombs(board,pos, all_bombs_strength, bomb_life)
        adjacent_flames = get_flames(board,pos)
        quarter = get_quarter(pos)
#        enemy_in_range = is_enemy_in_range(board,pos, blast_radius, enemies)
        thing_in_range = is_enemy_in_range(board,pos, blast_radius, enemies)
        
#        import IPython
#        IPython.embed()
        state = (valid_directions, dangerous_bombs, adjacent_flames,
                 can_kick, ammo, quarter, thing_in_range)
        return state
            
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
                self.episode_end();
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

    def episode_end(self):
        self.done = True
        self.last_action = 0
        self.new_action = 0
        self.cur_state = (0,)
        self.last_state = (0,)

