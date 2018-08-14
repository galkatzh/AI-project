'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
import numpy as np
from snorkel_agent import SnorkelAgent
from randoom_forest_agent import RandomForestAgent
from random import randint

def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent()
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    
    
    learner_index = randint(0,3)
    print(learner_index)
    agent_list[learner_index] = RandomForestAgent()
    
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    wins=np.zeros(4)
    episodes = 10
    
    win_str='winners'

    # Run the episodes just like OpenAI Gym
    for i_episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            steps += 1
            env.render()
            actions = env.act(state)
                
            state, reward, done, info = env.step(actions)
            
        if win_str in info.keys():
            for w in info[win_str]:
                if w==learner_index:
                    wins[w] += 1
        print('Episode {} finished'.format(i_episode))
    env.close()
    print('rates: ',wins/episodes)
    print('Learner win rate: ',wins[learner_index]/episodes)


if __name__ == '__main__':
    main()
