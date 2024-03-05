

from src.data_structures.sequence_mixer import *

def test_random_sequence_mixer():
        rsm = RandomSequenceMixer(np.ones(100),
                                  np.ones(100),
                                  np.ones(100),
                                  np.ones(100),
                                  np.ones(100))
        
        indiv, combined = rsm.get()

        assert len(indiv) == 5
        assert np.allclose(combined, np.full(100, 5))

        