from pommerman.agents import BaseAgent
import numpy as np
import random
import pommerman.utility as util
import pommerman.constants as consts
import os.path


filename = 'qvalues'
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

def powerup_in_range(board, pos):
    powers = [False] * 4
    inds = [[0,1],[2,3]]
    for i in range(-2,3):
        for j in range(-2,3):
            temp_pos = (pos[0] + i, pos[0] + j)
            if util.position_on_board(board, temp_pos) and util.position_is_powerup(board, temp_pos):
                powers[inds[i>=0][j>=0]] = True
                if i==0:
                    powers[inds[not i>=0][j>=0]] = True
                if j==0:
                    powers[inds[i>=0][not j>=0]] = True
    return tuple(powers)

def extract_state(obs):
        board = obs["board"]
        pos = np.array(obs["position"])
        bomb_life = obs["bomb_life"]
        all_bombs_strength = obs["bomb_blast_strength"]
        can_kick = obs['can_kick']
        blast_radius = obs['blast_strength']
        ammo = obs['ammo']
        enemies = obs['enemies']

        valid_directions = get_valid_directions(board, pos)
        dangerous_bombs = get_bombs(board,pos, all_bombs_strength, bomb_life)
        adjacent_flames = get_flames(board,pos)
        quarter = get_quarter(pos)
        enemy_in_range = is_enemy_in_range(board,pos, blast_radius, enemies)
        wood_in_range = is_wood_in_range(board,pos, blast_radius)
#        thing_in_range = is_enemy_in_range(board,pos, blast_radius, enemies)
        powerups = powerup_in_range(board,pos)

        state = (valid_directions, dangerous_bombs, adjacent_flames,
                 can_kick, ammo, quarter, wood_in_range, enemy_in_range,
                 powerups)
        return state


class ExtractedStateAgent(BaseAgent):

    def __init__(self, name,discount, c, alpha):
        super(ExtractedStateAgent, self).__init__()
        self.name = name
        self.last_action = 0
        self.new_action = 0
        self.cur_state = (0,)
        self.last_state = (0,)
#        self.epsilon = 0.8
#        self.discount = 1
#        self.alpha = 1
        self.c = c
        self.discount = discount
        self.alpha = alpha
        self.done = False
        self.q_values = dict()
        if os.path.isfile(self.get_filename()):
            self.q_values = np.load(self.get_filename())['q'].item()

    def get_filename(self):
        fn = "UCBqvalues_"
        fn += self.name
        fn += "_"
        fn += str(self.c)
        fn += "_"
        fn += str(self.alpha)
        fn += "_"
        fn += str(self.discount)
        fn += ".npz"
        return fn

    def save_qvalues(self):
        np.savez_compressed(self.get_filename(), q=self.q_values)

    def set_start_state(self, obs):
        self.cur_state = self.extract_state(obs)

    def update_q_value(self, reward, new_state = None, old_state = None, last_action=None):
        if new_state == None or old_state == None or last_action==None:
            if self.done:
                return
            if self.cur_state not in self.q_values:
                self.q_values[self.cur_state] = [np.zeros(6), np.zeros(6)]
            if self.last_state not in self.q_values:
                self.q_values[self.last_state] = [np.zeros(6), np.zeros(6)]
            best_next_action = np.argmax(self.q_values[self.cur_state][0])            
            td_target = reward + self.discount * self.q_values[self.cur_state][0][best_next_action]
            td_delta = td_target - self.q_values[self.last_state][0][self.last_action]
            self.q_values[self.last_state][0][self.last_action] += self.alpha * td_delta
            if reward != 0:
                self.episode_end(reward);
        else:
            if new_state not in self.q_values:
                self.q_values[new_state] = [np.zeros(6), np.zeros(6)]
            if old_state not in self.q_values:
                self.q_values[old_state] = [np.zeros(6), np.zeros(6)]
            best_next_action = np.argmax(self.q_values[new_state][0])
            td_target = reward + self.discount * self.q_values[new_state][0][best_next_action]
            td_delta = td_target - self.q_values[old_state][0][last_action]
            self.q_values[old_state][1][last_action] += 1
            self.q_values[old_state][0][last_action] += self.alpha * td_delta

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
        enemy_in_range = is_enemy_in_range(board,pos, blast_radius, enemies)
        wood_in_range = is_wood_in_range(board,pos, blast_radius)
#        thing_in_range = is_enemy_in_range(board,pos, blast_radius, enemies)
        stands_on_bomb = self.last_action == 5
        powerups = powerup_in_range(board,pos)

        state = (valid_directions, dangerous_bombs, adjacent_flames,
                 can_kick, ammo, quarter, wood_in_range, enemy_in_range,
                 stands_on_bomb, powerups)
        return state

    def act(self, obs, action_space):
        self.done = False
        self.last_action = self.new_action
        self.last_state = self.cur_state
        self.cur_state = self.extract_state(obs)

        if self.cur_state in self.q_values:
            if 0 in self.q_values[self.cur_state][1]: # if an action was never chosen, choose it
                actions_num = self.q_values[self.cur_state][1]
                self.new_action = np.random.choice(np.flatnonzero(actions_num == 0))
            else:
                conf = np.log(self.q_values[self.cur_state][1].sum()) / self.q_values[self.cur_state][1]
                bonus = self.c * np.sqrt(conf)
                self.new_action = np.argmax(self.q_values[self.cur_state][0] + bonus)
        else:
            self.new_action = action_space.sample()
            self.q_values[self.cur_state] = [np.zeros(6), np.zeros(6)]

        self.q_values[self.cur_state][1][self.new_action] += 1
        return self.new_action

    def episode_end(self, reward):
        if not self.done:
            self.done = True
            self.last_action = 0
            self.new_action = 0
            self.cur_state = (0,)
            self.last_state = (0,)
