import numpy as np


def distance_to_opponents(board, position, alive):
    my_num = board[position[0], position[1]]
    distances = []
    for agent in alive:
        if agent == my_num:
            continue
        x, y = np.where(board == agent)
        distances.append(abs(y - position[1]) + abs(x - position[0]))
