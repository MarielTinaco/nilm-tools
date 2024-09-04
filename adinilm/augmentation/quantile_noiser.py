
import numpy as np

from adinilm.filter.quantile_filter import quantile_filter

class NoisedInput:

        def noise(self, data, *args, **kwargs):
                _denoise = quantile_filter(data, 10, 50)
                data = data - np.percentile(data, 1)
                data = np.where(data < _denoise, _denoise, data)
                return quantile_filter(data, 10, 50)

        def __call__(self, data, *args, **kwargs):
                return self.noise(data, *args, **kwargs)