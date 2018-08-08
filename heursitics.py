import pommerman.utility as util
import pommerman.constants as consts
import snorkel
from snorkel.models import candidate_subclass
import numpy as np

def only_right(f):
    def inner(*args, **kwargs):
        return f(*args, **kwargs)[4]
    return inner


def no_flames(obs):
    res = [0]*6
    my_position = obs['position']
    board = obs['board']
    x, y = my_position
    for i,(a,b) in enumerate([(x, y+1), (x, y-1), (x-1, y), (x+1, y)]):
        if util.position_on_board(board, (a,b)) and board[a,b] == 4:
            res[i+1] = -1
    return res

def valid_directions(obs):
    res = [0]*6
    dirs = range(1,5)
    pos = obs['position']
    board = obs['board']
    res[1:5] = [1 if util.is_valid_direction(board, pos, d) else -1 for d in dirs]
    return res

def get_power_up(obs):
    res = [0] * 6
    my_position = obs['position']
    board = obs['board']
    x, y = my_position
    for i, (a, b) in enumerate([(x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)]):
        if util.position_on_board(board, (a,b)) and board[a, b] in [7,8,9]:
            res[i + 1] = 1
    return res

def get_lf():
    return [no_flames, valid_directions, get_power_up]