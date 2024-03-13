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


NormalizationProfiles = {
    NormalizationType.MINMAX: MinMaxScalling,
    NormalizationType.MODMINMAX: ModifiedMinMaxScalling,
    NormalizationType.STANDARD: StandardScalling,
}
