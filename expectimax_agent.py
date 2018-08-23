
from pommerman.agents import BaseAgent
import numpy as np
import random
from pommerman import utility
import pommerman.constants as consts
from omri_agent import *
from pommerman import forward_model as fm
from pommerman import characters
from pommerman import constants
import itertools


BOARD = 0
AGENTS = 1
BOMBS = 2
ITEMS = 3
FLAMES = 4

#levels in the expcetimax tree
MAX_LEVEL = 0
MEAN_LEVEL = 1

#expectimax tree depth
DEPTH = 2

def defualt_evaluation_function(state):
    return 0

#A state is a tuple:(curr_board, curr_agents, curr_bombs, curr_items, curr_flames)
class GameState:
    def __init__(self, board, agents_objects, bombs, items, flames):
        """
        Constructor
        :param board:
        :param agents:
        :param bombs:
        :param items:
        :param flames:
        """
        self.board = board
        self.agents_objects = agents_objects
        self.bombs = bombs
        self.items = items
        self.flames = flames


    def get_legal_actions(self):
        """
        :return: All possible combinations of legal actions of the agents.
        """
        positions = [ExpectimaxAgent.get_agent_position(self.board, utility.agent_value(self.agents_objects[i]._character.agent_id))
                     for i in range(len(self.agents_objects)) if self.agents_objects[i]._character.is_alive]
        actions_for_each_agent = []
        for pos in positions:
            actions_for_each_agent.append([dirc for dirc in range(5) if utility.is_valid_direction(self.board, pos, dirc)] +
                                          [constants.Action.Bomb.value])
        return list(itertools.product(*actions_for_each_agent))

    def generate_successor(self, actions):
        """
        :param agent_index:
        :param actions: list of actions, one for each agent
        :return: successor state of this state if each agent do its correspond action form the list
        """
        state = fm.ForwardModel.step(actions, self.board, self.agents_objects, self.bombs, self.items, self.flames)
        return GameState(state[BOARD], state[AGENTS], state[BOMBS], state[ITEMS], state[FLAMES])



#TODO: check how do we know agent's id
class ExpectimaxAgent(BaseAgent):
    def __init__(self, id, evaluation_function=defualt_evaluation_function):
        """
        Constructor
        :param evaluation_function:
        :param id
        """
        super(ExpectimaxAgent, self).__init__()
        self.evaluation_function = evaluation_function #this is a function that gets a state and return its value
        self.id = id
        self.env = None

    def set_env(self, env):
        self.env = env

    def act(self, obs, action_space):
        board = obs['board']
        bomb_life = obs["bomb_life"]
        all_bombs_strength = obs["bomb_blast_strength"]
        bombs = self.get_bombs(board, bomb_life, all_bombs_strength)
        flames = self.get_flames(board)
        #agents = obs['alive'] #TODO: check if obs["enemies"] is np array
        agents = self.env._agents
        items = self.get_items(board)
        state = GameState(board, agents, bombs, items, flames)
        return self.get_max(state, DEPTH)

    #TODO: check if get_bomb and get_flames should return np array
    @staticmethod
    def get_bombs(board, bombs_life, bombs_blast_strength):
        """
        :param board:
        :param bombs_life:
        :param bombs_blast_strength:
        :return: Bombs on the board
        """
        bombs = [characters.Bomb(None, (i, j), bombs_life[i][j], bombs_blast_strength[i][j])
                 for i in range(board.shape[0]) for j in range(board.shape[1]) if board[i][j] == constants.Item.Bomb.value]
        return bombs

    @staticmethod
    def get_flames(board):
        return [characters.Flame((i, j)) for i in range(board.shape[0]) for j in range(board.shape[1])
                if utility.position_is_flames(board, (i,j))]

    @staticmethod
    def get_items(board):
        """
        :param board:
        :return: A dictionary of positions and items
        """
        items_indexes = [constants.Item.ExtraBomb.value, constants.Item.IncrRange.value, constants.Item.Kick.value]
        items = dict()
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i][j] in items_indexes:
                    items[(i, j)] = board[i][j]
        return items

    @staticmethod
    def get_agent_position(board, agent_id):
        x, y = np.where(board == agent_id)
        return x[0], y[0] #TODO check this line


    def get_value(self, state):
        """
        Return value of a state
        :param state:
        :return:
        """
        return self.evaluation_function(state)

    def get_max(self, game_state, depth):
        """
        For max level in the tree.
        :param game_state:
        :param depth:
        :return:
        """
        if depth < 0:
            return
        """***When we need to return an action***"""
        if depth == DEPTH:
            acts = game_state.get_legal_actions()
            if len(acts) == 0 or depth == 0:
                return self.evaluation_function(game_state)

            max_arg = np.argmax([self.get_mean(game_state.generate_successor(action), depth - 1) \
                                 for action in game_state.get_legal_actions()])
            return acts[max_arg]
        """ELSE"""
        legals = game_state.get_legal_actions()
        if len(legals) == 0 or depth == 0:
            return self.evaluation_function(game_state)

        return max(self.get_mean(game_state.generate_successor(action), depth - 1) \
                   for action in game_state.get_legal_actions())

    def get_mean(self, game_state, depth):
        """
        Foe mean level in the tree.
        :param game_state:
        :param depth:
        :return:
        """
        if depth < 0:
            return

        legals = game_state.get_legal_actions()
        if len(legals) == 0:  # or depth == 0:
            return self.evaluation_function(game_state)

        return np.mean([self.get_max(game_state.generate_successor(action), depth) \
                        for action in game_state.get_legal_actions()])
