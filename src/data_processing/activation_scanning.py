
from abc import ABC, abstractmethod
from typing import Optional, Union, List
from numpy import ndarray


class ActivationScannerContext(object):

        def __init__(self, 
                        strategy : Optional[Union["ActivationScanner", str]] = None,
                        seq_len : int = 100,
                        *args,
                        **kwargs):

                if isinstance(strategy, str):
                        if strategy == "zero-padded":
                                strategy = ZeroPaddedActivationScanner(seq_len,
                                                                       stride=kwargs.get("num_windows") or 10)
                        else:
                                if strategy != "window":
                                        print(f"""
                                        WARNING: '{strategy}' mode not available. Please select between ["zero-padded", "window"].
                                        Defaulting to Window Activation scan mode.
                                        """)
                                strategy = WindowActivationScanner(seq_len,
                                                                        stride=kwargs.get("num_windows") or 10)

                self._strategy = strategy

        @property
        def seq_len(self):
                return self._strategy.seq_len

        @property
        def strategy(self):
                return self._strategy

        def __call__(self, data: ndarray) -> List[ndarray]:
                return self._strategy.scan(data)


class ActivationScanner(ABC):
    
        def __init__(self, seq_len, *args, **kwargs):
                self._seq_len = seq_len

        @property
        def seq_len(self):
               return self._seq_len

        @abstractmethod
        def scan(self, data: ndarray) -> List[ndarray]:
               raise NotImplementedError


class ZeroPaddedActivationScanner(ActivationScanner):

        def __init__(self, seq_len, stride, *args, **kwargs):
                super(ZeroPaddedActivationScanner, self).__init__(seq_len, *args, **kwargs)
                self.stride = stride

        def scan(self, data: ndarray) -> List[ndarray]:
                if data.shape[0] <= self.seq_len:
                        return [data]


class WindowActivationScanner(ActivationScanner):
       
       def __init__(self, seq_len, stride, *args, **kwargs):
              super(WindowActivationScanner, self).__init__(seq_len, *args, **kwargs)
              self.stride = stride

       def scan(self, data: ndarray) -> List[ndarray]:
              ...