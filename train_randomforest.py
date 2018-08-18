import pickle
from sklearn.ensemble import RandomForestClassifier
import pommerman
from pommerman import agents
import numpy as np
from heursitics import get_lf
from snorkel.learning import GenerativeModel
from extracted_state_agent import extract_state
from scipy import sparse



def load_snorkel():
    filename = 'snorkel_model'
    gms = []
    for i in range(6):
        gm = GenerativeModel()
        gm.load(filename + str(i))
        gms.append(gm)
    return gms


def merge(obs):
    res = []
    for t in obs:
        if isinstance(t, tuple):
            res += list(t)
        else:
            res.append(t)
    return np.array(res)



def main():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''
    
    filename = "rf.pickle"
    
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
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    gms = load_snorkel()
    # Run the episodes just like OpenAI Gym
    train_states = []
    train_labels = []
    for i_episode in range(100):
        state = env.reset()
        done = False
        while not done:
            #            env.render()
            cur_obs = env.get_observations()
            actions = env.act(state)
            for ob in cur_obs:
                train_states.append(merge(extract_state(ob)))
                probs = np.zeros(6)
                l = np.array([f(ob) for f in get_lf()])
                for i,m in enumerate(gms):
                    tmp = sparse.csr_matrix(l[:,i])
                    probs[i] = m.marginals(tmp)
                train_labels.append(probs)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))
    env.close()


    train_labels = np.array([np.array(list(map(int,prob == prob.max()))) for prob in train_labels])
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(train_states, train_labels)

    with open(filename, 'wb') as handle:
        pickle.dump(rf, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    main()
