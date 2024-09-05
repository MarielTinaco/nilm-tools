
import numpy as np
import random

class RandomSequenceMixer(object):
    
        def __init__(self, *args, **kwargs):
                self.args = args

        def get(self):
                ret = [random.choice(i) for i in self.args]
                combined = np.sum(ret, axis=0)
                return ret, combined
