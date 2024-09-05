__all___ = ["BinarizationContext", "ThresholdBinarization", "binarization"]

import numpy as np
from enum import Enum
from typing import Optional, Union, List
from abc import ABC, abstractmethod

class BinarizationTypes(Enum):
        THRESHOLD = "threshold"

class BinarizationContext(object):
    
        def __init__(self, 
                     strategy : Optional[Union["Binarization", str]] = None,
                     *args, **kwargs):
                

                if isinstance(strategy, str):
                        if BinarizationTypes(strategy) == BinarizationTypes.THRESHOLD:
                                strategy = ThresholdBinarization(threshold=kwargs.get("threshold") or 10)

                        else:
                                print(f"""
                                WARNING: '{strategy}' mode not available. Please select between ["threshold"].
                                Defaulting to Threshold Binarization mode.
                                """)
                                strategy = ThresholdBinarization(threshold=kwargs.get("threshold") or 10)

                self._strategy = strategy

        def __call__(self, data: np.ndarray) -> np.ndarray:
                return self._strategy.binarize(data=data)


class Binarization(ABC):

        @abstractmethod
        def binarize(self, data: np.ndarray) -> np.ndarray:
                raise NotImplementedError


class ThresholdBinarization(Binarization):

        def __init__(self, threshold, *args, **kwargs):
                self._threshold = threshold

        
        def binarize(self, data: np.ndarray) -> np.ndarray:
                """[summary]
                
                Arguments:
                        data {[type]} -- [description]
                        threshold {[type]} -- [description]
                
                Returns:
                        [type] -- [description]
                """
                return np.where(data>= self._threshold,1,0).astype(int)
        

def binarization(data, mode="threshold", **kwargs):

        if mode not in BinarizationTypes._member_names_:
                mode = "threshold"

        if mode == "threshold":
                thresh = kwargs.get("threshold")
        
        binarizer = BinarizationContext(strategy=mode, threshold=thresh)

        return binarizer(data)