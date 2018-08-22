'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
import numpy as np
#from snorkel_agent import SnorkelAgent
#from randoom_forest_agent import RandomForestAgent
from random import randint
from extracted_state_agent import ExtractedStateAgent
from mcts_agent import MCTSAgent

def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    episodes = 2
    # Create a set of agents (exactly four)
    for i_episode in range(episodes):
        agent_list = [
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent()
            # agents.DockerAgent("pommerman/simple-agent", port=12345),
        ]
        
        
        learner_index = randint(0,3)
        print(learner_index)
        agent_list[learner_index] = MCTSAgent() #ExtractedStateAgent('extract', 0, 0, 0)
        
        
    #    agent_list[learner_index] = RandomForestAgent()
        #agent_list[learner_index] = SnorkelAgent()
        
        agent_list[learner_index].set_agent_id(learner_index)
    #    agent_list[learner_index].epsilon = 0
        # Make the "Free-For-All" environment using the agent list
        env = pommerman.make('PommeFFACompetition-v0', agent_list)
#        env.set_training_agent(learner_index)
        wins=np.zeros(4)
        
        
        win_str='winners'

    # Run the episodes just like OpenAI Gym
    
        obs = env.reset()
        state = env.get_json_info()
        agent_list[learner_index].set_state(state)
        done = False
        steps = 0
        while not done and steps < 500:
            steps += 1
            # env.render()
            #actions = env.act(state)
            actions = env.act(obs)
#            action = agent_list[learner_index].search(state)
#            actions.insert(learner_index, action)
            obs, step_reward, done, info = env.step(actions)
            state = env.get_json_info()
            agent_list[learner_index].set_state(state)
            print(actions)
                
            

        print(info)
        if win_str in info.keys():
            for w in info[win_str]:
                wins[w] += 1
        print('Episode {} finished'.format(i_episode))
        env.close()
    print('rates: ',wins/episodes)
    print('Learner win rate: ',wins[learner_index]/episodes)


if __name__ == '__main__':
    main()
