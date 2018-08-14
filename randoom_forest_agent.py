import numpy as np
import pickle
from extracted_state_agent import extract_state
from pommerman.agents import BaseAgent

def merge(obs):
    res = []
    for t in obs:
        if isinstance(t, tuple):
            res += list(t)
        else:
            res.append(t)
    return np.array(res)

class RandomForestAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(RandomForestAgent, self).__init__(*args, **kwargs)
        
        filename = "rf.pickle"
        
        with open(filename, 'rb') as handle:
            self.rf = pickle.load(handle)


    def act(self, obs, action_space):
        obs = merge(extract_state(obs))
        probs = self.rf.predict_proba([obs])[0] 

        return np.random.choice(np.flatnonzero(probs == probs.max()))

