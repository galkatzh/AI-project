import pommerman
from pommerman import agents
from extracted_state_agent import ExtractedStateAgent
from random import randint
import sys
import json
import numpy as np

DEBUG = False

def main():
    
    WINDOW_SIZE = 5
    WINDOW_SHIFT = 5

    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    discount = 1
    alpha = 1
    epsilon = 0.5
    if len(sys.argv) == 4:
        epsilon = float(sys.argv[1])
        alpha = float(sys.argv[2])
        discount = float(sys.argv[3])
    elif len(sys.argv) == 3:
        epsilon = float(sys.argv[1])
        alpha = float(sys.argv[2])
        
    bla = ExtractedStateAgent("backplay", discount, epsilon, alpha)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        #        agents.RandomAgent(),
        #agents.RandomAgent(),
        #        bla
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    learner_index = randint(0,3)
    print(learner_index)
    agent_list[learner_index] = bla
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeTeamCompetition-v0', agent_list)
    reward = [0,0,0,0]

    # Run the episodes just like OpenAI Gym
    for i_episode in range(50):
        learning_dicts = []
        game_snapshots = []


        state = env.reset()
        #        import IPython
        #        IPython.embed()
        cur_obs = env.get_observations()
        bla.set_start_state(cur_obs[learner_index])
        done = False
        while not done:
            #                input()
            game_snapshots.append(env.get_json_info())
            actions = env.act(state)

            state, reward, done, info = env.step(actions)

            last_obs = cur_obs
            cur_obs = env.get_observations()
            learning_dicts.append([{'reward': reward[i], 'new_state': bla.extract_state(cur_obs[i]),
                                        'old_state': bla.extract_state(last_obs[i]), 'last_action':actions[i]} for i in range(4)])

            #bla.update_q_value(reward[learner_index])

            if DEBUG and reward[learner_index] == 0:
                env.render()
                print(bla.cur_state)
            #                input()

            #print(reward, done, info)
            #        print(bla.q_values)
        learning_dicts = learning_dicts[::-1]
        T = len(game_snapshots)
        j = 0
        k = WINDOW_SIZE
        while T-k < len(game_snapshots) and T-j < len(game_snapshots) - WINDOW_SIZE:
            with open('game_state.json', 'w') as outfile:
                json.dump(np.random.choice(game_snapshots[T-j:min([len(game_snapshots)-1,T-k])]), outfile)
            env.set_init_game_state('game_state.json')
            env.set_json_info()
            cur_obs = env.get_observations()
            bla.set_start_state(cur_obs[learner_index])
            done = False
            while not done:
                actions = env.act(state)
                state, reward, done, info = env.step(actions)
                last_obs = cur_obs
                cur_obs = env.get_observations()
                for i in range(4):
                    bla.update_q_value(reward[i],new_state = bla.extract_state(cur_obs[i]),
                                        old_state = bla.extract_state(last_obs[i]), last_action=actions[i])
            j += WINDOW_SHIFT
            k += WINDOW_SHIFT
        print('Episode {} finished'.format(i_episode))
    bla.save_qvalues()
    env.close()


if __name__ == '__main__':
    main()