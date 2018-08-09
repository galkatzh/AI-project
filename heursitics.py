import pommerman.utility as util
import pommerman.constants as consts
import numpy as np
import queue
from collections import defaultdict

dirs = [consts.Action.Up, consts.Action.Down, consts.Action.Left, consts.Action.Right]


def no_flames(obs):
    res = [0]*6
    my_position = obs['position']
    board = obs['board']
    x, y = my_position
    for act in dirs:
        next_pos  = util.get_next_position(my_position, act)
        if util.position_on_board(board, next_pos) and util.position_is_flames(board, next_pos):
            res[act.value] = -1
    return res

def valid_directions(obs):
    res = [0]*6
    pos = obs['position']
    board = obs['board']
    enemies = obs['enemies']
    for act in dirs:
        next_pos  = util.get_next_position(pos, act)
        if util.position_on_board(board, next_pos) and util.position_is_passable(board, next_pos, enemies):
            res[act.value] = 1
        else:
            res[act.value] = -1
    return res

def get_power_up(obs):
    res = [0] * 6
    my_position = obs['position']
    board = obs['board']
    for act in dirs:
        next_pos  = util.get_next_position(my_position, act)
        if util.position_on_board(board, next_pos) and util.position_is_powerup(board,next_pos):
            res[act.value] = 1
    return res


def convert_bombs(bomb_map):
    ret = []
    locations = np.where(bomb_map > 0)
    for r, c in zip(locations[0], locations[1]):
        ret.append({
            'position': (r, c),
            'blast_strength': int(bomb_map[(r, c)])
        })
    return ret


def unsafe_directions(obs):
    my_position = tuple(obs['position'])
    board = np.array(obs['board'])
    bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
    enemies = [consts.Item(e) for e in obs['enemies']]
    items, dist, prev = _djikstra(board, my_position, bombs, enemies, depth=10)
    unsafe_directions = _directions_in_range_of_bomb(board, my_position, bombs, dist)
    res = [0] * 6
    for key in unsafe_directions:
        if unsafe_directions[key] > 0:
            res[key.value] = -1
    return res


def not_stuck_directions(obs):
    my_position = tuple(obs['position'])
    board = np.array(obs['board'])
    enemies = [consts.Item(e) for e in obs['enemies']]

    def is_stuck_direction(next_position, bomb_range, next_board, enemies):
        Q = queue.PriorityQueue()
        Q.put((0, next_position))
        seen = set()

        nx, ny = next_position
        is_stuck = True
        while not Q.empty():
            dist, position = Q.get()
            seen.add(position)

            px, py = position
            if nx != px and ny != py:
                is_stuck = False
                break

            if dist > bomb_range:
                is_stuck = False
                break

            for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_position = (row + px, col + py)
                if new_position in seen:
                    continue

                if not util.position_on_board(next_board, new_position):
                    continue

                if not util.position_is_passable(next_board,
                                                    new_position, enemies):
                    continue

                dist = abs(row + px - nx) + abs(col + py - ny)
                Q.put((dist, new_position))
        return is_stuck

    res = [0] * 6

    next_board = board.copy()
    next_board[my_position] = consts.Item.Bomb.value

    for direction in dirs:
        next_position = util.get_next_position(
            my_position, direction)
        nx, ny = next_position
        if not util.position_on_board(next_board, next_position) or \
                not util.position_is_passable(next_board, next_position, enemies):
            continue

        if not is_stuck_direction(next_position, obs['bomb_blast_strength'][nx, ny], next_board, enemies):
            # We found a direction that works. The .items provided
            # a small bit of randomness. So let's go with this one.
            res[direction.value] = 1
    if res == [0] * 6:
        res = [-1] * 6
        res[0] = 1
    return res




    # Lay pomme if we are adjacent to an enemy.


def kill_enemy(obs):
    my_position = tuple(obs['position'])
    board = np.array(obs['board'])
    bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
    enemies = [consts.Item(e) for e in obs['enemies']]
    ammo = int(obs['ammo'])
    blast_strength = int(obs['blast_strength'])
    items, dist, prev = _djikstra(board, my_position, bombs, enemies, depth=10)
    res = [0] * 6
    if _is_adjacent_enemy(items, dist, enemies) and _maybe_bomb(
            ammo, blast_strength, items, dist, my_position):
        res[-1] = 1
    return res


    # Move towards an enemy if there is one in exactly three reachable spaces.


def move2enemy(obs):
    res = [0] * 6
    my_position = tuple(obs['position'])
    board = np.array(obs['board'])
    bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
    enemies = [consts.Item(e) for e in obs['enemies']]
    items, dist, prev = _djikstra(board, my_position, bombs, enemies, depth=10)
    direction = _near_enemy(my_position, items, dist, prev, enemies, 3)
    if direction:
        res[direction.value] = 1
    return res


    # Maybe lay a bomb if we are within a space of a wooden wall.


def wooden_wall(obs):
    res = [0] * 6
    my_position = tuple(obs['position'])
    board = np.array(obs['board'])
    bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
    enemies = [consts.Item(e) for e in obs['enemies']]
    ammo = int(obs['ammo'])
    blast_strength = int(obs['blast_strength'])
    items, dist, prev = _djikstra(board, my_position, bombs, enemies, depth=10)
    if _near_wood(my_position, items, dist, prev, 1):
        if _maybe_bomb(ammo, blast_strength, items, dist, my_position):
            res[-1] = 1
    return res


    # Move towards a wooden wall if there is one within two reachable spaces and you have a bomb.


def move2wooden(obs):
    res = [0] * 6
    my_position = tuple(obs['position'])
    board = np.array(obs['board'])
    bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
    enemies = [consts.Item(e) for e in obs['enemies']]
    items, dist, prev = _djikstra(board, my_position, bombs, enemies, depth=10)
    direction = _near_wood(my_position, items, dist, prev, 2)
    if direction:
        res[direction.value] = 1
    return res


def is_dead_end(board, position, direction):
    npos = util.get_next_position(position, direction)
    counter = 0
    for d in dirs:
        if not util.is_valid_direction(board, npos, d):
            counter += 1
    return counter >= 3


def dead_end(obs):
    res = [0] * 6
    pos = tuple(obs['position'])
    board = np.array(obs['board'])
    for direction in dirs:
        next_pos  = util.get_next_position(pos, direction)
        if util.position_on_board(board, next_pos) and is_dead_end(board, pos, direction):
            res[direction.value] = -1
    return res


def _djikstra(board, my_position, bombs, enemies, depth=None, exclude=None):
    assert (depth is not None)

    if exclude is None:
        exclude = [
            consts.Item.Fog, consts.Item.Rigid, consts.Item.Flames
        ]

    def out_of_range(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return depth is not None and abs(y2 - y1) + abs(x2 - x1) > depth

    items = defaultdict(list)
    dist = {}
    prev = {}
    Q = queue.PriorityQueue()

    mx, my = my_position
    for r in range(max(0, mx - depth), min(len(board), mx + depth)):
        for c in range(max(0, my - depth), min(len(board), my + depth)):
            position = (r, c)
            if any([
                out_of_range(my_position, position),
                util.position_in_items(board, position, exclude),
            ]):
                continue

            if position == my_position:
                dist[position] = 0
            else:
                dist[position] = np.inf

            prev[position] = None
            Q.put((dist[position], position))

    for bomb in bombs:
        if bomb['position'] == my_position:
            items[consts.Item.Bomb].append(my_position)

    while not Q.empty():
        _, position = Q.get()

        if util.position_is_passable(board, position, enemies):
            x, y = position
            val = dist[(x, y)] + 1
            for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_position = (row + x, col + y)
                if new_position not in dist:
                    continue

                if val < dist[new_position]:
                    dist[new_position] = val
                    prev[new_position] = position

        item = consts.Item(board[position])
        items[item].append(position)

    return items, dist, prev


def _directions_in_range_of_bomb(board, my_position, bombs, dist):
    ret = defaultdict(int)

    x, y = my_position
    for bomb in bombs:
        position = bomb['position']
        distance = dist.get(position)
        if distance is None:
            continue

        bomb_range = bomb['blast_strength']
        if distance > bomb_range:
            continue

        if my_position == position:
            # We are on a bomb. All directions are in range of bomb.
            for direction in [
                consts.Action.Right,
                consts.Action.Left,
                consts.Action.Up,
                consts.Action.Down,
            ]:
                ret[direction] = max(ret[direction], bomb['blast_strength'])
        elif x == position[0]:
            if y < position[1]:
                # Bomb is right.
                ret[consts.Action.Right] = max(
                    ret[consts.Action.Right], bomb['blast_strength'])
            else:
                # Bomb is left.
                ret[consts.Action.Left] = max(ret[consts.Action.Left],
                                                 bomb['blast_strength'])
        elif y == position[1]:
            if x < position[0]:
                # Bomb is down.
                ret[consts.Action.Down] = max(ret[consts.Action.Down],
                                                 bomb['blast_strength'])
            else:
                # Bomb is down.
                ret[consts.Action.Up] = max(ret[consts.Action.Up],
                                               bomb['blast_strength'])
    return ret




def _is_adjacent_enemy(items, dist, enemies):
    for enemy in enemies:
        for position in items.get(enemy, []):
            if dist[position] == 1:
                return True
    return False


def _has_bomb(obs):
    return obs['ammo'] >= 1


def _maybe_bomb(ammo, blast_strength, items, dist, my_position):
    """Returns whether we can safely bomb right now.

    Decides this based on:
    1. Do we have ammo?
    2. If we laid a bomb right now, will we be stuck?
    """
    # Do we have ammo?
    if ammo < 1:
        return False

    # Will we be stuck?
    x, y = my_position
    for position in items.get(consts.Item.Passage):
        if dist[position] == np.inf:
            continue

        # We can reach a passage that's outside of the bomb strength.
        if dist[position] > blast_strength:
            return True

        # We can reach a passage that's outside of the bomb scope.
        px, py = position
        if px != x and py != y:
            return True

    return False


def _nearest_position(dist, objs, items, radius):
    nearest = None
    dist_to = max(dist.values())

    for obj in objs:
        for position in items.get(obj, []):
            d = dist[position]
            if d <= radius and d <= dist_to:
                nearest = position
                dist_to = d

    return nearest


def _get_direction_towards_position(my_position, position, prev):
    if not position:
        return None

    next_position = position
    while prev[next_position] != my_position and prev[next_position] is not None:
        next_position = prev[next_position]

    return util.get_direction(my_position, next_position)

    
def _near_enemy(my_position, items, dist, prev, enemies, radius):
    nearest_enemy_position = _nearest_position(dist, enemies, items, radius)
    return _get_direction_towards_position(my_position,
                                           nearest_enemy_position, prev)




def _near_wood(my_position, items, dist, prev, radius):
    objs = [consts.Item.Wood]
    nearest_item_position = _nearest_position(dist, objs, items, radius)
    return _get_direction_towards_position(my_position,
                                           nearest_item_position, prev)



def get_lf():
    return [no_flames, valid_directions, get_power_up, unsafe_directions, 
            not_stuck_directions, kill_enemy, move2enemy, wooden_wall,
            move2wooden, dead_end]