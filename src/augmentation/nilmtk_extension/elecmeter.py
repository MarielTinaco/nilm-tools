import numpy as np
import nilmtk

from typing import Protocol


class ActivationExtension(Protocol):
    
        def extend(self, *args, **kwargs) -> np.ndarray:
                pass


class ElecmeterActivationAppender:

        def __init__(self, data : nilmtk.ElecMeter):
                self.data = data

        def _generator(self, activations, interval=0):
                for activation in activations:
                        stacked = np.hstack([activation.values, np.zeros(interval)])
                        yield stacked

        def extend(self, num_samples = 0, interval=0) -> np.ndarray:
                power_series = self.data.power_series_all_data()
                acts = np.hstack(list(self._generator(self.data.get_activations(), interval=interval or ())))
                return np.hstack([power_series.values, acts[:num_samples]])