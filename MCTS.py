import numpy as np
import time

import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants

AMOUNT_ACTIONS = 6
EXPLORATION_CONST = 1
NUM_CHECKS = 5
GAMMA = 0.1
EPSILON = 1e-10


"""
POMME questions:
Do we save the qvalues every time?
Do we want to do live search?
How do we incorporate POMCP?(follow the algorithm) whats the purpose of o when stepping to the next?
incorporate NN because otherwise it will probably be shit.
"""

def simulate_action(env, my_action, my_agents_id):
    actions = env.act(env.get_observations())
    actions.insert(my_agents_id, my_action)
    new_obs, rewards, done, info = env.step(actions)
    reward = rewards[my_agents_id]
    new_state = env.get_json_info()
    return None


class Node(object):

    """MCTS Node"""

    def __init__(self):
        self.q_vals = np.zeros(AMOUNT_ACTIONS)
        self.n_visits = np.zeros(AMOUNT_ACTIONS)
        self.rewards = np.zeros(AMOUNT_ACTIONS)
        self.probs = None

    def __init__(self, probs):
        self.q_vals = np.zeros(AMOUNT_ACTIONS)
        self.n_visits = np.zeros(AMOUNT_ACTIONS)
        self.rewards = np.zeros(AMOUNT_ACTIONS)
        self.probs = probs

    def next_action(self):
        U = EXPLORATION_CONST * np.sqrt(np.sum(self.n_visits)) / (1 + self.n_visits) #so we wont divide by zero
        values = self.q_vals + U
        max_val = np.max(values)
        indices = np.where(values == max_val)
        return np.random.choice(indices)

    def update_node(self, action, reward):
        assert AMOUNT_ACTIONS > action >= 0

        self.n_visits[action] += 1
        self.rewards[action] += reward
        self.q_vals[action] = self.rewards[action] / self.n_visits[action]


class MCTSAgent(BaseAgent):
    def act(self, obs, action_space):
        state_str = str(obs[self.agent_id])
        if state_str in self.tree:
            np.argmax(self.tree[state_str].qvals)
        else:
            print("we randomming")
            np.random.choice(action_space)

    def __init__(self, agent_id):
        super().__init__()
        self.agent_id = agent_id
        self.env = self.make_env()
        self.tree = {}

    def make_env(self):
        agents = []
        for agent_id in range(4):
            if agent_id == self.agent_id:
                agents.append(self)
            else:
                agents.append(SimpleAgent())

        return pommerman.make('PommeFFACompetition-v0', agents)

    def search(self, state):
        for i in range(NUM_CHECKS):
            self.simulate(state ,0)
        obs = state.get_observations()

        return np.argmax(self.tree[str(obs[self.agent_id])].qvals)

    def simulate(self, state, depth):
        if GAMMA ** depth < EPSILON:
            return 0

        self.env._init_game_state = state
        obs = self.env.get_observations()

        state_str = str(obs[self.agent_id])

        if state_str not in self.tree:
            self.tree[state_str] = Node()
            return self.rollout(state, depth)

        best_action = self.tree[state_str].next_action()
        """observe the env and act"""
        actions = self.env.act(obs)
        actions.insert(self.agent_id, best_action)
        new_obs, rewards, done, info = self.env.step(actions)
        reward = rewards[self.agent_id]
        new_state = self.env.get_json_info()

        #update reward + terminate if we're done
        total_reward = reward if done else reward + GAMMA * self.simulate(new_state, depth + 1)
        """ end observation, we have a reward"""
        self.tree[state_str].update_node(best_action, total_reward)

        #get the curr state to be the original one
        self.env._init_game_state = state
        self.env.reset()

        return total_reward

    def rollout(self, state, depth):
        if GAMMA ** depth < EPSILON:
            return 0

        self.env._init_game_state = state
        obs = self.env.get_observations()

        action = self.policy_roll(state)
        """observe the env and act"""
        actions = self.env.act(obs)
        actions.insert(self.agent_id, action)
        new_obs, rewards, done, info = self.env.step(actions)
        reward = rewards[self.agent_id]
        next_state = self.env.get_json_info()
        """ end observation, we have a reward"""
        total_reward = reward + GAMMA * self.rollout(next_state, depth + 1)
        self.env._init_game_state = state
        self.env.reset()

        return total_reward
