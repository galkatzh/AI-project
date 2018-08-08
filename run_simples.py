'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
import numpy as np
from snorkel_agent import SnorkelAgent

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
        SnorkelAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    d=[]

    # Run the episodes just like OpenAI Gym
    for i_episode in range(10):
        state = env.reset()
        done = False
        while not done:
#            env.render()
            cur_obs = env.get_observations()
            actions = env.act(state)
            for ob, act in zip(cur_obs, actions):
                val = np.zeros(6)
                val[act] = 1
                d.append((ob,val))
                
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))
    env.close()
    np.savez_compressed('labels_foe_snorkel',d=d)


if __name__ == '__main__':
    main()
