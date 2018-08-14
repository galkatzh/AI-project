import pommerman
from pommerman import agents
from extracted_state_agent import ExtractedStateAgent
from random import randint
import sys

DEBUG = False

def main():

    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    discount = 1;
    if len(sys.argv) != 3:
        alpha = 1
        epsilon = 0.5
    else:
        epsilon = float(sys.argv[1])
        alpha = float(sys.argv[2])
    bla = ExtractedStateAgent(discount, epsilon, alpha)

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



        state = env.reset()
        #        import IPython
        #        IPython.embed()
        cur_obs = env.get_observations()
        bla.set_start_state(cur_obs[learner_index])
        done = False
        while not done:
            #                input()
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
        for lst in learning_dicts:
            for player_i in range(4):
                bla.update_q_value(**lst[player_i])
        print('Episode {} finished with rewards: {}'.format(i_episode, reward))
    bla.save_qvalues()
    env.close()


if __name__ == '__main__':
    main()