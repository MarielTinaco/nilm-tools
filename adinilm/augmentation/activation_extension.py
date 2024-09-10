from __future__ import annotations

import nilmtk
import numpy as np
from abc import ABC
from enum import Enum, auto

from .nilmtk_extension.elecmeter import ElecmeterActivationAppender, ElecmeterActivationRandomizer, \
                NILMTKActivationExtenderRegistry, NILMTKActivationExtenderTypes

class ActivationExtensionTypes(Enum):
        ZERO = "zero"

class ActivationExtensionContext(object):
    
        def __init__(self, strategy : "ActivationExtensionStrategy"):

                self.strategy = strategy

        def extend(self, *args, **kwargs):
                return self.strategy.extend(*args, **kwargs)


class ActivationExtensionStrategy(ABC):

        def extend(self, *args, **kwargs) -> np.ndarray:
                raise NotImplementedError

class ZeroActivationExtension(ActivationExtensionStrategy):

        def __init__(self, data : np.ndarray):
                self.data = data
        
        def extend(self, num_samples = 0, *args, **kwargs) -> np.ndarray:
                return np.hstack([self.data, np.zeros(num_samples)])


ActivationExtensionRegistry = {
        ActivationExtensionTypes.ZERO : ZeroActivationExtension
}

def extend_activations(data, num_full_samples, mode=None, *args, **kwargs):
        
        if hasattr(data, "power_series_all_data"):
                assert num_full_samples > len(data.power_series_all_data()), \
                        f"Select num_full_samples value greated than {len(data.power_series_all_data())}"

                if mode:
                        ActivationExtension = NILMTKActivationExtenderRegistry[NILMTKActivationExtenderTypes(mode)]
                else:
                        ActivationExtension = ElecmeterActivationAppender

                num_samples = int(num_full_samples - len(data.power_series_all_data()))
                extras = {"interval": kwargs.get("interval")}

        else:
                ActivationExtension = ActivationExtensionRegistry(ActivationExtensionTypes(mode))
                num_samples = int(num_full_samples - len(data))

                extras = {}

        ctx = ActivationExtensionContext(ActivationExtension(data))
        return ctx.extend(num_samples=num_samples, **extras)



if __name__ == "__main__":
        import matplotlib.pyplot as plt

        import nilm.utils.paths_manager as pathsman

        ## NILMTK Example 

        BUILDING = 1
        TIME_START = "2015-01-01"
        TIME_END = "2015-01-15"
        APPLIANCE = "washer dryer"

        dataset = nilmtk.DataSet(pathsman.UKDALE_H5_PATH)
        dataset.set_window(start=TIME_START, end=TIME_END)
        elecmeter : nilmtk.ElecMeter = dataset.buildings[BUILDING].elec[APPLIANCE]

        # strat = ElecmeterActivationAppender(data=elecmeter)
        # ctx = ActivationExtensionContext(strategy=strat)
        # extended = ctx.extend(num_samples=2000, interval=50)

        extended = extend_activations(data=elecmeter, num_full_samples=200000, mode="randomizer", interval=10)

        # print(len(extended))

        plt.plot(extended[-3001:-1])
        plt.show()