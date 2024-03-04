__all___ = ["SequenceScannerContext", "WindowSequenceScanner"]

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, List


class SequenceScannerContext(object):

        DEFAULT_SCAN_NUM = 2

        def __init__(self, 
                        strategy : Optional[Union["SequenceScanner", str]] = None,
                        seq_len : int = 100,
                        *args,
                        **kwargs):

                if strategy is None:
                        strategy = WindowSequenceScanner(seq_len,
                                                           n_windows=kwargs.get("num_windows") or self.DEFAULT_SCAN_NUM)

                elif isinstance(strategy, str):
                        if strategy != "window":
                                print(f"""
                                WARNING: '{strategy}' mode not available. Please select between ["window"].
                                Defaulting to Window Sequence scan mode.
                                """)
                                strategy = WindowSequenceScanner(seq_len,
                                                                n_windows=kwargs.get("num_windows") or self.DEFAULT_SCAN_NUM)

                self._strategy = strategy

        @property
        def seq_len(self):
                return self._strategy.seq_len

        @property
        def strategy(self):
                return self._strategy

        def __call__(self, data: np.ndarray) -> List[np.ndarray]:
                return self._strategy.scan(data)


class SequenceScanner(ABC):
    
        def __init__(self, seq_len, *args, **kwargs):
                self._seq_len = seq_len

        @property
        def seq_len(self):
               return self._seq_len

        @abstractmethod
        def scan(self, data: np.ndarray) -> List[np.ndarray]:
               raise NotImplementedError


class WindowSequenceScanner(SequenceScanner):
       
        def __init__(self, seq_len, n_windows, *args, **kwargs):
              super(WindowSequenceScanner, self).__init__(seq_len, *args, **kwargs)
              self.n_windows = n_windows

        def scan(self, data: np.ndarray) -> List[np.ndarray]:
                if data.shape[0] == self.seq_len:
                        return [data]
                elif data.shape[0] < self.seq_len:
                        pads = self.seq_len - data.shape[0]
                        left_pad = pads//2
                        right_pad = pads - left_pad
                        res = np.pad(data, (left_pad, right_pad), 'constant', constant_values=0)
                        return [res]

                diff = int((data.shape[0]//self.n_windows)-1)
                container = []
                for n in range(self.n_windows):
                        start = int(n*diff)
                        end = start + self.seq_len
                        if end + 1 <= data.shape[0]:
                                container.append(data[start:end])
                return container


        def __repr__(self) -> str:
               return "window"