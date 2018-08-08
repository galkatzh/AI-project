import numpy as np
from snorkel.learning import GenerativeModel
from scipy import sparse
from pommerman.agents import BaseAgent
from heursitics import get_lf

filename = 'snorkel_model'

class SnorkelAgent(BaseAgent):
    
    def __init__(self, *args, **kwargs):
        super(SnorkelAgent, self).__init__(*args, **kwargs)
        #TODO: load model
#        self.models = np.load(filename)['m'].item()
        gms = []
        for i in range(6):
            gm = GenerativeModel()
            gm.load(filename + str(i))
            gms.append(gm)
        self.models = gms

    def act(self, obs, action_space):
        probs = np.zeros(6)
        
        l = np.array([f(obs) for f in get_lf()])
        
        for i,m in enumerate(self.models):
            tmp = sparse.csr_matrix(l[:,i])
            probs[i] = m.marginals(tmp)
        
        return np.argmax(probs)
            
        
