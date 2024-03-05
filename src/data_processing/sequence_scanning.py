__all___ = ["SequenceScannerContext", "WindowSequenceScanner" , "OddWindowSequenceScanner"]

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union, List

class SequenceScannerType(Enum):
        WINDOW = "window"
        ODD_WINDOW = "odd_window"


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
                        if SequenceScannerType(strategy) == SequenceScannerType.WINDOW:
                                strategy = WindowSequenceScanner(seq_len,
                                                                 n_windows=kwargs.get("num_windows") or self.DEFAULT_SCAN_NUM)                                
                        elif SequenceScannerType(strategy) == SequenceScannerType.ODD_WINDOW:
                                strategy = OddWindowSequenceScanner(seq_len)       

                        else:
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

        def generator(self, data):
                return self._strategy.generator(data=data)

        def __call__(self, data: np.ndarray) -> List[np.ndarray]:
                return self._strategy.scan(data)


class SequenceScanner(ABC):
    
        def __init__(self, seq_len, *args, **kwargs):
                self._seq_len = seq_len

        @property
        def seq_len(self):
               return self._seq_len

        @abstractmethod
        def generator(self, data):
                raise NotImplementedError

        @abstractmethod
        def scan(self, data: np.ndarray) -> List[np.ndarray]:
               raise NotImplementedError


class WindowSequenceScanner(SequenceScanner):
       
        def __init__(self, seq_len, n_windows, *args, **kwargs):
              super(WindowSequenceScanner, self).__init__(seq_len, *args, **kwargs)
              self.n_windows = n_windows

        def scan(self, data: np.ndarray) -> List[np.ndarray]:
                return list(self.generator(data))

        def generator(self, data: np.ndarray) -> List[np.ndarray]:
                if data.shape[0] == self.seq_len:
                        yield data
                elif data.shape[0] < self.seq_len:
                        pads = self.seq_len - data.shape[0]
                        left_pad = pads//2
                        right_pad = pads - left_pad
                        res = np.pad(data, (left_pad, right_pad), 'constant', constant_values=0)
                        yield res
                else:
                        diff = int((data.shape[0]//self.n_windows)-1)

                        for n in range(self.n_windows):
                                start = int(n*diff)
                                end = start + self.seq_len
                                if end < data.shape[0]:
                                        yield data[start:end]

        def __repr__(self) -> str:
               return "window"


class OddWindowSequenceScanner(SequenceScanner):

        def scan(self, data: np.ndarray) -> List[np.ndarray]:
                return list(self.generator(data))

        def generator(self, data):
                seq_len = self.seq_len - 1 if self.seq_len % 2==0 else self.seq_len
                units_to_pad = seq_len // 2
                new_mains = np.pad(data, (units_to_pad,units_to_pad),'constant',constant_values=(0,0))  
                for i in range(len(new_mains) - seq_len+1):
                        yield new_mains[i:i + seq_len]


def scan_sequences(data, seq_len=100, num_windows=20, mode="window", *args, **kwargs):

        if mode not in SequenceScannerType._member_names_:
                mode = "window"

        scanner = SequenceScannerContext(strategy=mode,
                                         seq_len=seq_len,
                                         num_windows=num_windows)

        return scanner(data)