import numpy as np
import time
import multiprocessing
import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants

AMOUNT_ACTIONS = 6
EXPLORATION_CONST = 1
NUM_CHECKS = 300
GAMMA = 0.9
EPSILON = 1e-10
NUM_AGENTS = 4

NUM_EPISODES = 512
NUM_RUNNERS = 8

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

    # def __init__(self, probs):
    #     self.q_vals = np.zeros(AMOUNT_ACTIONS)
    #     self.n_visits = np.zeros(AMOUNT_ACTIONS)
    #     self.rewards = np.zeros(AMOUNT_ACTIONS)
    #     self.probs = probs

    def next_action(self):
        U = EXPLORATION_CONST * np.sqrt(np.sum(self.n_visits)) / (1 + self.n_visits) #so we wont divide by zero
        values = self.q_vals + U
        max_val = np.max(values)
        indices = np.array(np.where(values == max_val)).reshape(-1,)
        return np.random.choice(indices)

    def update_node(self, action, reward):
        assert AMOUNT_ACTIONS > action >= 0

        self.n_visits[action] += 1
        self.rewards[action] += reward
        self.q_vals[action] = self.rewards[action] / self.n_visits[action]


class MCTSAgent(BaseAgent):

    def policy_roll(self, state):
        return self.policy_fcn(state)

    def act(self, obs, action_space):
        state_str = str(obs)
        if state_str in self.tree:
            np.argmax(self.tree[state_str].q_vals)
        else:
            print("we randomming")
            np.random.choice(action_space)

    def __init__(self, *args, **kwargs):
        super(MCTSAgent, self).__init__(*args, **kwargs)
        self.agent_id = -1
        self.env = self.make_env()
        self.policy_fcn = lambda x: np.random.choice(6)
        self.tree = {}

    def set_agent_id(self, agent_id):
        self.agent_id=agent_id

    def make_env(self):
        agents = []
        for agent_id in range(4):
            if agent_id == self.agent_id:
                agents.append(self)
            else:
                agents.append(SimpleAgent())
        env = pommerman.make('PommeFFACompetition-v0', agents)
        env.set_training_agent(self.agent_id)
        return env

    def search(self, state):
        for i in range(NUM_CHECKS):
            self.simulate(state ,0)
        self.env._init_game_state = state
        obs = self.env.reset()

        return np.argmax(self.tree[str(obs[self.agent_id])].q_vals)

    def simulate(self, state, depth):
        if GAMMA ** depth < EPSILON:
            return 0

        self.env._init_game_state = state
        obs = self.env.reset()

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
        obs = self.env.reset()

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
def runner(id, num_episodes, fifo, _args):
    # make sure agents play at all positions
    agent_id = id % NUM_AGENTS


    for j in range(num_episodes):
        agent_list = []
        agent = MCTSAgent()
        for i in range(NUM_AGENTS):
            if i == agent_id:
                agent.set_agent_id(agent_id)
                agent_list.append(agent)
            else:
                agent_list.append(SimpleAgent())
        print(agent_list)
        env = pommerman.make('PommeFFACompetition-v0', agent_list)
        env.set_training_agent(agent_id)
        step = 0
        # Run the episodes just like OpenAI Gym
        sum_rewards = 0
        obs = env.reset()
        state = env.get_json_info()
        done = False
        start_time = time.time()
        while not done:
            # env.render()
            actions = env.act(obs)
            action = agent.search(state)
            actions.insert(agent_id, action)
            obs, step_reward, done, info = env.step(actions)
            state = env.get_json_info()
            sum_rewards += step_reward[agent_id]
            step += 1
            env.save_json("./agent"+str(id)+"_episode_"+str(j)+"_")

        elapsed = time.time() - start_time
        env.close()
        fifo.put((step, sum_rewards, agent_id, elapsed))


if __name__ == "__main__":
    # runner(0, 1, None, None)
    assert NUM_EPISODES % NUM_RUNNERS == 0, "The number of episodes should be divisible by number of runners"

    # use spawn method for starting subprocesses
    ctx = multiprocessing.get_context('spawn')

    # create fifos and processes for all runners
    fifo = ctx.Queue()
    for i in range(NUM_RUNNERS):
        process = ctx.Process(target = runner, args=(i, NUM_EPISODES // NUM_RUNNERS, fifo, None))
        process.start()

    # do logging in the main process
    all_rewards = []
    all_lengths = []
    all_elapsed = []
    for i in range(NUM_EPISODES):
        # wait for a new trajectory
        length, rewards, agent_id, elapsed = fifo.get()

        print("Episode:", i, "Length:", length, "Rewards:", rewards, "Agent:", agent_id, "Time per step:", elapsed / length)
        all_rewards.append(rewards)
        all_lengths.append(length)
        all_elapsed.append(elapsed)

    print("Average reward:", np.mean(all_rewards))
    print("Average length:", np.mean(all_lengths))
    print("Time per timestep:", np.sum(all_elapsed) / np.sum(all_lengths))
