import pommerman
import numpy as np
from extracted_state_agent import ExtractedStateAgent
from random_forest_agent import RandomForestAgent
from mcts_agent import MCTSAgent
from ucb_extracted_state_agent import UCBExtractedStateAgent
from snorkel_agent import SnorkelAgent
from all_state_agent import NewAgent


import sys

DEBUG = False


def run_games(agent_construct_list):
    wins = np.zeros(2)
    win_str='winners'
    episodes = 50

    for i_episode in range(episodes):
        agent_list = [agent_construct_list[i]() for i in range(len(agent_construct_list))]
        for agent in agent_list:
            agent.epsilon = 0
        for agent_index in range(len(agent_list)):
            if isinstance(agent_list[agent_index], MCTSAgent):
                agent_list[agent_index].set_agent_id(agent_index)
        # Make the "Free-For-All" environment using the agent list
        env = pommerman.make('PommeFFACompetition-v0', agent_list)
        obs = env.reset()
        state = env.get_json_info()
        for agent_index in range(len(agent_list)):
            if isinstance(agent_list[agent_index], MCTSAgent):
                agent_list[agent_index].set_state(state)
        done = False
        steps = 0
        while not done and steps < 800:
            steps += 1
            actions = env.act(obs)
            obs, reward, done, info = env.step(actions)
            state = env.get_json_info()
            for agent_index in range(len(agent_list)):
                if isinstance(agent_list[agent_index], MCTSAgent):
                    agent_list[agent_index].set_state(state)
            if DEBUG:
                env.render()
        if win_str in info.keys():
            for w in info[win_str]:
                if w in [0, 2]:
                    wins[0] += 1
                if w in [1, 3]:
                    wins[1] += 1
        print('Episode {} finished'.format(i_episode))
        env.close()
    return wins


q_params = dict()
ucb_params = dict()

def get_backplay():
    return ExtractedStateAgent(name = "backplay", **q_params)

def get_extract():
    return ExtractedStateAgent(name = "extract", **q_params)

def get_ucb():
    return UCBExtractedStateAgent(name = "extract", **ucb_params)

def get_full_state():
    return NewAgent(name = "full", **q_params)



all_constuctors = [RandomForestAgent, SnorkelAgent, MCTSAgent, get_backplay, get_extract, get_full_state, get_ucb]

def main():
    idx = [int(sys.argv[i]) for i in range(1, len(sys.argv))]
    idx = sorted(idx)
    print(idx)
    game_res = np.zeros((len(idx), len(idx)))
    game_count = np.zeros((len(idx), len(idx)))
    agents = [all_constuctors[i] for i in idx]
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i != j:
                curr_agent_list = [agents[i], agents[j], agents[i], agents[j]]
                curr_wins = run_games(curr_agent_list)
                game_res[i,j] += curr_wins[0]
                game_res[j,i] += curr_wins[1]
                game_count[i,j] += curr_wins[0] + curr_wins[1]
                game_count[j,i] += curr_wins[0] + curr_wins[1]
    name = "_".join([str(i) for i in idx])
    final = game_res / game_count
    np.save("wins_" +name , game_res)
    np.save("games_" + name, game_count)
    np.save("win_ratio_" + name, final)



if __name__ == '__main__':
    main()
