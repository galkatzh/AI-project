import pommerman
from pommerman import agents
from random import randint
#from random_forest_agent import RandomForestAgent
from snorkel_agent import SnorkelAgent




def main():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)
    wins = 0
    games = 0
    win_str='winners'
    episodes = 100
    for i_episode in range(episodes):
        # Create a set of agents (exactly four)
        agent_list = [
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            # agents.DockerAgent("pommerman/simple-agent", port=12345),
        ]

        learner_index = randint(0,3)
        print(learner_index)
    #    agent_list[learner_index] = RandomForestAgent()
        agent_list[learner_index] = SnorkelAgent()

    # Make the "Free-For-All" environment using the agent list
        env = pommerman.make('PommeFFACompetition-v0', agent_list)
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            steps += 1
            #env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        if win_str in info.keys():
            games += 1
            for w in info[win_str]:
                if w == learner_index:
                    wins += 1
        print('Episode {} finished'.format(i_episode))
        env.close()
    print('rates: ',wins/games)
    print(wins, ",", games)



if __name__ == '__main__':
    main()
