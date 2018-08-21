import pommerman
import numpy as np
from extracted_state_agent import ExtractedStateAgent
import copy
import sys


alpha_list = [0.4, 0.2, 0.1, 0.01]
epsilon_list = [0.5, 0.2, 0.1, 0.05]
discount_list = [0.8, 0.9, 0.99, 0.999]


def run_games(param_dict1, param_dict2):
    wins = np.zeros(2)
    win_str='winners'
    episodes = 25

    for i_episode in range(episodes):
        agent_list = [ExtractedStateAgent(**param_dict1), ExtractedStateAgent(**param_dict2),
                      ExtractedStateAgent(**param_dict1), ExtractedStateAgent(**param_dict2)]
        for agent in agent_list:
            agent.epsilon = 0
        # Make the "Free-For-All" environment using the agent list
        env = pommerman.make('PommeTeamCompetition-v0', agent_list)
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            steps += 1
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        if win_str in info.keys():
            for w in info[win_str]:
                if w == 0:
                    wins[0] += 1
                if w == 1:
                    wins[1] += 1
        print('Episode {} finished'.format(i_episode))
        env.close()
    return wins


def all_pairs(param_list, key, param_dict):
    all_params = []
    for p in param_list:
        param = copy.deepcopy(param_dict)
        param[key] = p
        all_params.append(param)
    print(all_params)
    all_wins = np.zeros(len(all_params))
    all_games = np.zeros(len(all_params))
    for i in range(len(all_params)):
        for j in range(len(all_params)):
            if i != j:
                curr_wins = run_games(all_params[i], all_params[j])
                all_wins[i] += curr_wins[0]
                all_wins[j] += curr_wins[1]
                all_games[i] = curr_wins[0] + curr_wins[1]
                all_games[j] = curr_wins[0] + curr_wins[1]
    return all_wins/all_games



def main():
    run_type = 0
    if len(sys.argv) == 2:
        run_type = int(sys.argv[1])

    if run_type == 1:
        free_discount = np.zeros((4,4,4))
        for i, epsilon in enumerate(epsilon_list):
            for j, alpha in enumerate(alpha_list):
                param = {'name': 'extract', 'discount': None, 'epsilon': epsilon, 'alpha': alpha}
                free_discount[i,j,:] = all_pairs(discount_list, "discount", param)
        np.save("free_discount", free_discount)

    elif run_type == 2:
        free_alpha = np.zeros((4,4,4))
        for i, epsilon in enumerate(epsilon_list):
            for j, discount in enumerate(discount_list):
                param = {'name': 'extract', 'discount': discount, 'epsilon': epsilon, 'alpha': None}
                free_alpha[i,j,:] = all_pairs(alpha_list, "alpha", param)
        np.save("free_alpha", free_alpha)

    elif run_type == 3:
        free_epsilon = np.zeros((4,4,4))
        for i, alpha in enumerate(alpha_list):
            for j, discount in enumerate(discount_list):
                param = {'name': 'extract', 'discount': discount, 'epsilon': None, 'alpha': alpha}
                free_epsilon[i,j,:] = all_pairs(epsilon_list, "epsilon", param)
        np.save("free_epsilon", free_epsilon)




if __name__ == '__main__':
    main()
