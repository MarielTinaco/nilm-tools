__all___ = ["MinMaxScalling", "ModifiedMinMaxScaling", "StandardScaling"]

from abc import ABC, abstractmethod
import numpy as np
from enum import Enum


class NormalizationType(Enum):
    MINMAX = "minmax"
    MODMINMAX = "mod_minmax"
    STANDARD = "standard"


class NormalizationContext(object):
    def __init__(self, strategy: "NormalizeStrategy"):
        self.strategy = strategy

    def normalize(self, data):
        return self.strategy.normalize(data)

    def denormalize(self, data):
        return self.strategy.denormalize(data)


class NormalizeStrategy(ABC):
    @abstractmethod
    def normalize(self, data) -> np.ndarray:
        pass

    @abstractmethod
    def denormalize(self, data) -> np.ndarray:
        pass


class MinMaxScaling(NormalizeStrategy):
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalize(self, data) -> np.ndarray:
        norm_data = (data - self.min) / (self.max - self.min)
        return norm_data

    def denormalize(self, data) -> np.ndarray:
        denorm_data = data * (self.max_val - self.min_val) + self.min_val
        return denorm_data


class ModifiedMinMaxScaling(NormalizeStrategy):
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalize(self, data) -> np.ndarray:
        norm_data = 2 * ((data - self.min) / (self.max - self.min)) - 1
        return norm_data

    def denormalize(self, data) -> np.ndarray:
        denorm_data = (((data + 1) * (self.max - self.min)) / 2) + self.min
        return denorm_data


class StandardScaling(NormalizeStrategy):
    def __init__(self, mean_val, std_val):
        self.std = std_val
        self.mean = mean_val

    def normalize(self, data) -> np.ndarray:
        norm_data = (data - self.mean) / self.std
        return norm_data

    def denormalize(self, data) -> np.ndarray:
        denorm_data = (data * self.std) + self.mean
        return denorm_data


NormalizationRegistry = {
    NormalizationType.MINMAX: MinMaxScaling,
    NormalizationType.MODMINMAX: ModifiedMinMaxScaling,
    NormalizationType.STANDARD: StandardScaling,
}


class NormalizationHandler(object):

    def __init__(self, mode, *args, **kwargs):
        #additional guard clause for different normalization

        self.select_mode(mode, *args, **kwargs)

#add function for switch mode
    def select_mode(self, mode, *args, **kwargs):
        if mode not in [i.value for i in NormalizationType]:
            raise f"{mode} Handling Not Supported"
            
        Handler = NormalizationRegistry[NormalizationType(mode)]
        self.ctx = NormalizationContext(strategy=Handler(*args, **kwargs))


    def normalize(self, data):
        return self.ctx.normalize(data)

    def denormalize(self, data):
        return self.ctx.denormalize(data)

# if __name__ == "__main__" :
#     test = NormalizationHandler(mode="minmax", min_val = 100, max_val=500)
#     test.normalize(data)