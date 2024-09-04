
from typing import Callable
import numpy as np

import adik

def test_sequence_sampler():
        data = np.random.rand(1000)
        length = 100
        sampler = adik.SequenceSampler(data=data, length=length)

        assert sampler.axis == 0
        assert isinstance(sampler.indices, np.ndarray)

        samp1 = sampler[0]

        assert samp1.shape[0] == length


def test_1xN_sequence_sampler():
        N = 5
        data = np.random.rand(1000, N)

        length = 1
        axis = 0

        class test_stride:

                def __init__(self, start, stop, step):
                        self.indices = np.arange(start=start, stop=stop, step=step)
                
                def __call__(self, data):
                        return self.indices

        stride = test_stride(100, 1000, 1)

        sampler = adik.SequenceSampler(data=data, length=length,
                                       axis=axis, stride=stride)

        samp1 = sampler[0]

        assert samp1.shape[0] == length
        assert samp1.shape[1] == N