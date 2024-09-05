from adinilm.data_processing.normalize import NormalizationHandler
import numpy as np

def test_normalize():
    ## dummy data creation
    dummy = np.random.randint(100, size=(100))
    test_norm = NormalizationHandler(mode="minmax", min_val=dummy.min(), max_val=dummy.max())
    normalized_dummy = test_norm.normalize(dummy)

    assert normalized_dummy.max() == 1
    assert normalized_dummy.min() == 0