import pommerman
from pommerman import agents
import numpy as np
from heursitics import get_lf
from snorkel.learning import GenerativeModel

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
        agents.SimpleAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeTeamCompetition-v0', agent_list)
    d=[]

    # Run the episodes just like OpenAI Gym
    for i_episode in range(300):
        state = env.reset()
        done = False
        while not done:
#            env.render()
            cur_obs = env.get_observations()
            actions = env.act(state)
            for ob, act in zip(cur_obs, actions):
                val = np.zeros(6)
                val[act] = 1
                d.append([ob,val])
                
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))
    env.close()
    
    
    lf = get_lf()
    
    rows = len(d)
    
    L = np.zeros([6,rows,len(lf)])
    for r in range(rows):
        for i,f in enumerate(lf):
            L[:,r,i] = f(d[r][0])
            
    gms = []
    for i in range(6):
        gms.append(GenerativeModel())


    # TODO: add ground labels to training
    filename = 'snorkel_model'
    for i,gm in enumerate(gms):
        temp_l = np.squeeze(L[i,:,:]).astype(int)
        gm.train(temp_l)
        gm.save(filename + str(i))
        


if __name__ == '__main__':
    main()
