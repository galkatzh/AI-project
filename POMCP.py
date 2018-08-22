import numpy as np
import time
import multiprocessing
import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants

AMOUNT_ACTIONS = 6
EXPLORATION_CONST = 1
NUM_CHECKS = 5
GAMMA = 0.1
EPSILON = 1e-10

NUM_AGENTS = 4

NUM_EPISODES = 4
NUM_RUNNERS = 4
NUM_ITERS = 100

N_PARTICLES = 4

"""
POMME questions:
Do we save the qvalues every time?
Do we want to do live search?
How do we incorporate POMCP?(follow the algorithm) whats the purpose of o when stepping to the next?
incorporate NN because otherwise it will probably be shit.
"""


def random_choice(some_arr):
    return some_arr[np.random.choice(len(some_arr))]


class POMCPNode(object):

    """POMCP Node"""

    def __init__(self):
        self.values = np.zeros(AMOUNT_ACTIONS)
        self.n_visits = np.zeros(AMOUNT_ACTIONS)
        self.particles = []
        self.probs = None

    def next_action(self):
        vals = self.values + EXPLORATION_CONST * np.sqrt(np.log(np.sum(self.n_visits))/self.n_visits)
        max_val = vals.max()
        indices = np.where(vals == max_val)
        return np.random.choice(indices)

    def update_node(self, action, reward):
        assert AMOUNT_ACTIONS > action >= 0

        self.n_visits[action] += 1
        self.values[action] += (reward - self.values[action]) / self.n_visits


class POMCPAgent(BaseAgent):
    def generate_particles(self, history, num_particles):
        self.env.reset()
        prev_state = self.env.get_json_info()
        particles = []
        str_init_history = str(history[:-2])
        while len(particles) < num_particles:
            si = random_choice(self.tree[str_init_history].particles)
            self.env._init_game_state = si
            obs = self.env.reset()
            actions = self.env.act(obs)
            action = history[-2]
            actions.insert((self.agent_id, action))
            sj, oj, r, cost = self.env.step(actions)

            if oj == obs:
                particles.append(sj)
        #TODO: add invigoration
        return particles

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

    def search(self, history):
        str_history = str(history)
        for i in range(NUM_CHECKS):
            if len(history) == 0:
                self.env._init_game_state = None
                state = self.env.get_json_info()
            else:
                #TODO: make sure there are enough particles. see: pyPOMDP pomcp.update_belief
                ### BEGIN ATTEMPT ###
                if str_history not in self.tree:
                    self.tree[str_history] = POMCPNode()
                    self.tree[str_history].particles.extend(self.generate_particles(history, N_PARTICLES))
                ###

                state = random_choice(self.tree[str_history].particles)

            self.simulate(state, history, 0)

        return np.argmax(self.tree[str_history].values)

    def simulate(self, state, history, depth):
        if GAMMA ** depth < EPSILON:
            return 0

        self.env._init_game_state = state
        obs = self.env.get_observations()

        history_str = str(history)

        if history_str not in self.tree:
            self.tree[history_str] = POMCPNode()
            return self.rollout(state, history, depth)

        best_action = self.tree[history_str].next_action()
        """observe the env and act"""
        actions = self.env.act(obs)
        actions.insert(self.agent_id, best_action)
        new_obs, rewards, done, info = self.env.step(actions)
        reward = rewards[self.agent_id]
        new_state = self.env.get_json_info()

        #update reward + terminate if we're done
        history.extend([best_action, new_obs[self.agent_id]])
        total_reward = reward if done else reward + GAMMA * self.simulate(new_state, history, depth + 1)
        """ end observation, we have a reward"""
        self.tree[history_str].update_node(best_action, total_reward)

        #get the curr state to be the original one
        self.env._init_game_state = state
        self.env.reset()

        return total_reward

    def rollout(self, state, history, depth):
        if GAMMA ** depth < EPSILON:
            return 0

        self.env._init_game_state = state
        obs = self.env.get_observations()

        action = self.policy_roll(history)
        """observe the env and act"""
        actions = self.env.act(obs)
        actions.insert(self.agent_id, action)
        new_obs, rewards, done, info = self.env.step(actions)
        reward = rewards[self.agent_id]
        next_state = self.env.get_json_info()

        history.extend([action, new_obs[self.agent_id]])
        """ end observation, we have a reward"""
        total_reward = reward + GAMMA * self.rollout(next_state, history,depth + 1)
        self.env._init_game_state = state
        self.env.reset()

        return total_reward


def runner(id, num_episodes, fifo, _args):
    # make sure agents play at all positions
    agent_id = id % NUM_AGENTS
    agent = POMCPAgent(agent_id=agent_id)
    agent_list = []

    for i in range (NUM_AGENTS):
        if i == agent_id:
            agent_list.append(SimpleAgent())
        else:
            agent_list.append(SimpleAgent())

    for j in range(num_episodes):
        print(agent_list)
        env = pommerman.make('PommeTeamCompetition-v0', agent_list)
        env.set_training_agent(agent_id)
        step = 0
        # Run the episodes just like OpenAI Gym
        sum_rewards = 0
        state = env.reset()
        done = False
        start_time = time.time()
        print(state)
        history = [env.get_observations()[agent_id]]
        while not done:
            # env.render()
            actions = env.act(state)
            action = agent.search(history)
            actions.insert((agent_id, action))
            state, step_reward, done, info = env.step(actions)

            history.extend([action,env.get_observations()[agent_id]])

            sum_rewards += step_reward[agent_id]
            step += 1
        elapsed = time.time() - start_time
        env.close()
        # fifo.put((step, sum_rewards, agent_id, elapsed))


if __name__ == "__main__":
    runner(0, 1, None, None)
    # assert NUM_EPISODES % NUM_RUNNERS == 0, "The number of episodes should be divisible by number of runners"
    #
    # # use spawn method for starting subprocesses
    # ctx = multiprocessing.get_context('spawn')
    #
    # # create fifos and processes for all runners
    # fifo = ctx.Queue()
    # for i in range(NUM_RUNNERS):
    #     process = ctx.Process(target = runner, args=(i, NUM_EPISODES // NUM_RUNNERS, fifo, None))
    #     process.start()
    #
    # # do logging in the main process
    # all_rewards = []
    # all_lengths = []
    # all_elapsed = []
    # for i in range(NUM_EPISODES):
    #     # wait for a new trajectory
    #     length, rewards, agent_id, elapsed = fifo.get()
    #
    #     print("Episode:", i, "Length:", length, "Rewards:", rewards, "Agent:", agent_id, "Time per step:", elapsed / length)
    #     all_rewards.append(rewards)
    #     all_lengths.append(length)
    #     all_elapsed.append(elapsed)
    #
    # print("Average reward:", np.mean(all_rewards))
    # print("Average length:", np.mean(all_lengths))
    # print("Time per timestep:", np.sum(all_elapsed) / np.sum(all_lengths))