import random
import numpy as np
import itertools as it
import nilmtk


from enum import Enum
from typing import Protocol, Union, Callable
from concurrent.futures import ThreadPoolExecutor


class NILMTKActivationExtenderTypes(Enum):
        APPENDER = "appender"
        RANDOMIZER = "randomizer"
        RIGHTPADDER = "rightpadder"


class ActivationExtension(Protocol):
    
        def extend(self, *args, **kwargs) -> np.ndarray:
                pass

class ElecmeterActivationRandomizer:

        def __init__(self, data : nilmtk.ElecMeter):
                self.data = data

        def _generator(self, activations, interval_func : Callable):
                while True: yield np.hstack([random.choice(activations).values, np.zeros(interval_func())])

        def extend(self, num_samples = 0, interval : Union[Callable, int] = 0):
                interval_func = interval if isinstance(interval, Callable) else lambda : interval
                num_samples = num_samples + len(self.data.power_series_all_data())
                activation_stack = []
                accumulated_length = 0

                for activation in self._generator(self.data.get_activations(), interval_func=interval_func):
                        if accumulated_length > num_samples:
                                break

                        activation_stack.append(activation)
                        accumulated_length += len(activation)

                acts = np.hstack(activation_stack)
                return acts[:num_samples]


class ElecmeterActivationAppender:

        def __init__(self, data : nilmtk.ElecMeter):
                self.data = data

        def _generator(self, activations, interval_func : Callable):
                while True: yield np.hstack([random.choice(activations).values, np.zeros(interval_func())])

        def extend(self, num_samples = 0, interval : Union[Callable, int] = 0) -> np.ndarray:
                power_series = self.data.power_series_all_data()
                interval_func = interval if isinstance(interval, Callable) else lambda : interval
                activation_stack = []
                accumulated_length = 0

                for activation in self._generator(self.data.get_activations(), interval_func=interval_func):
                        if accumulated_length > num_samples:
                                break
                        activation_stack.append(activation)
                        accumulated_length += len(activation)

                acts = np.hstack(activation_stack)
                return np.hstack([power_series.values, acts[:num_samples]])


class ElecmeterActivationRightPadder:

        def __init__(self, data : nilmtk.ElecMeter):
                self.data = data

        def extend(self, num_samples = 0, padding_mode : Union[Callable, int, str] = 0, **kwargs) -> np.ndarray:
                power_series = self.data.power_series_all_data()
                if isinstance(padding_mode, str):
                        mode = padding_mode
                        padding_kwargs = kwargs
                elif isinstance(padding_mode, int):
                        mode = 'constant'
                        padding_kwargs = {'constant_values' : (padding_mode, padding_mode)}
                elif isinstance(padding_mode, Callable):
                        mode = padding_mode
                        padding_kwargs = kwargs
                else:
                        mode = 'empty'
                        padding_kwargs = kwargs

                return np.pad(array=power_series.values, 
                              pad_width=(0, int(num_samples)),
                              mode=mode,
                              **padding_kwargs)


NILMTKActivationExtenderRegistry = {
        NILMTKActivationExtenderTypes.APPENDER : ElecmeterActivationAppender,
        NILMTKActivationExtenderTypes.RANDOMIZER : ElecmeterActivationRandomizer,
        NILMTKActivationExtenderTypes.RIGHTPADDER : ElecmeterActivationRightPadder,
}