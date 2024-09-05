
from typing import Optional, Union, Callable, Protocol
import numpy as np


class WindowSampler:
    
        def __init__(self, data : np.ndarray, stride : Optional[Union[int, np.ndarray, Callable]] = None,  *args, **kwargs):
                """
                
                Parameters
                :param data:
                :type  data: numpy array

                Keyword arguments
                :param length: Length of sequence along axis
                :type  length: int
                :param axis: Axis of traversal
                :type  axis:
                """

                self._length = int(kwargs.get("length", 1))
                self._axis = int(kwargs.get("axis", 0))

                if len(data.shape) == 1:
                        self._axis = 0

                indexable = data.shape[self._axis] - self._length

                if isinstance(stride, int):
                        self.indices = np.arange(0, indexable, stride or 1)
                elif isinstance(stride, np.ndarray):
                        self.indices = stride[stride < indexable]
                elif isinstance(stride, Callable):
                        self.indices = stride(data)
                        self.indices = self.indices[self.indices < indexable]
                else:
                        self.indices = np.arange(0, indexable, 1)
        
                assert len(self.indices.shape) == 1, "Stride/Indexing must be 1D data"

                self.data = data


        def __len__(self):
                return len(self.indices)


        def __getitem__(self, index):
                # Optimized numpy.take
                ptr = self.indices[index]
                idx = [slice(None)]*self.data.ndim
                idx[self._axis] = range(ptr, ptr+self.length)
                return self.data[tuple(idx)]


        @property
        def length(self):
                return self._length
        
        @property
        def axis(self):
                return self._axis



if __name__ == "__main__":

        data = np.random.rand(1000)

        samp = SequenceSampler(data=data, length=100)

        print(samp[0])