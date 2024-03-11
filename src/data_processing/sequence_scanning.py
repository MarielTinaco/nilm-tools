__all___ = ["SequenceScannerContext", "WindowSequenceScanner" , "OddWindowSequenceScanner", "SlidingShortWindowSequenceScanner",
            "LengthAdaptiveSequenceScanner"]

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union, List

class SequenceScannerType(Enum):
        WINDOW = "window"
        ODD_WINDOW = "odd_window"
        SLIDING_SHORT_WINDOW = "slide_short"
        ADAPTIVE = "adaptive"
        TAIL = "tail"


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
                        strategy = seqScannerFactory[SequenceScannerType(strategy)](seq_len=seq_len, \
                                                                                    n_windows = kwargs.get("num_windows") or self.DEFAULT_SCAN_NUM)

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
    
        def __init__(self, seq_len, n_windows=None, *args, **kwargs):
                self._seq_len = seq_len
                self.n_windows = n_windows

        @property
        def seq_len(self):
               return self._seq_len

        def scan(self, data: np.ndarray) -> List[np.ndarray]:
                return list(self.generator(data))

        @abstractmethod
        def generator(self, data):
                raise NotImplementedError


class WindowSequenceScanner(SequenceScanner):

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
               return SequenceScannerType.WINDOW.value


class OddWindowSequenceScanner(SequenceScanner):

        def generator(self, data):
                seq_len = self.seq_len - 1 if self.seq_len % 2==0 else self.seq_len
                units_to_pad = seq_len // 2
                new_mains = np.pad(data, (units_to_pad,units_to_pad),'constant',constant_values=(0,0))  
                for i in range(len(new_mains) - seq_len+1):
                        yield new_mains[i:i + seq_len]

        def __repr__(self) -> str:
               return SequenceScannerType.ODD_WINDOW.value


class SlidingShortWindowSequenceScanner(SequenceScanner):

        def generator(self, data):
                width = data.shape[0]
                assert self.seq_len >= width, "Sequence length must be greater than or equal data width"
                max_initial_position = self.seq_len - width
                strides = max_initial_position // self.n_windows
                for i in range(self.n_windows):
                        left_pad = int(i*strides)
                        right_pad = self.seq_len - left_pad - width
                        padded = np.pad(data, (left_pad, right_pad), 'constant', constant_values=(0, 0))
                        yield padded

        def __repr__(self) -> str:
               return SequenceScannerType.SLIDING_SHORT_WINDOW.value


class LengthAdaptiveSequenceScanner(SequenceScanner):

        def generator(self, data):
                width = data.shape[0]
                if self.seq_len >= width:
                        return SlidingShortWindowSequenceScanner(self.seq_len, self.n_windows).generator(data)
                else:
                        return WindowSequenceScanner(self.seq_len, self.n_windows).generator(data)

        def __repr__(self) -> str:
               return SequenceScannerType.ADAPTIVE.value


class TailendSequenceScanner(SequenceScanner):

        def __init__(self, seq_len, n_windows=None, *args, **kwargs):
                self._seq_len = seq_len
                self.n_windows = n_windows

                if isinstance(self.n_windows, int):
                        self._left_tail_windows = self.n_windows // 2
                        self._right_tail_windows = self.n_windows - self._left_tail_windows
                elif isinstance(self.n_windows, tuple):
                        assert len(self.n_windows) == 2, "Must declare number windows for both left and right tail"
                        self._left_tail_windows = self.n_windows[0]
                        self._right_tail_windows = self.n_windows[1]
                else:
                        raise ValueError(f"n_windows parameter doesn't support {type(self.n_windows)}")

        def generator(self, data):
                width = data.shape[0]
                assert self.seq_len <= width, "Sequence length must be less than or equal data width"
                
                for i in range(self._left_tail_windows):
                        strides = width // (self._left_tail_windows)
                        left_pad = int(i*strides)
                        padded = np.pad(data, (left_pad, 0), 'constant', constant_values=(0, 0))
                        yield padded[:self.seq_len]

                for i in range(self._right_tail_windows):
                        strides = width // (self._right_tail_windows)
                        right_pad = int(i*strides)
                        padded = np.pad(data, (0, right_pad), 'constant', constant_values=(0, 0))
                        yield padded[-self.seq_len-1:-1]


def scan_sequences(data, seq_len=100, num_windows=20, mode="window", *args, **kwargs):
        if mode not in [i.value for i in SequenceScannerType]:
                mode = "window"
                print(f"""
                        WARNING: '{mode}' mode not available. Please select between {SequenceScannerType._member_names_}.
                        Defaulting to Window Sequence scan mode.
                """)

        scanner = SequenceScannerContext(strategy=mode,
                                         seq_len=seq_len,
                                         num_windows=num_windows)

        return scanner(data)


seqScannerFactory = {
        SequenceScannerType.WINDOW : WindowSequenceScanner,
        SequenceScannerType.ODD_WINDOW : OddWindowSequenceScanner,
        SequenceScannerType.SLIDING_SHORT_WINDOW : SlidingShortWindowSequenceScanner,
        SequenceScannerType.ADAPTIVE : LengthAdaptiveSequenceScanner,
        SequenceScannerType.TAIL : TailendSequenceScanner
}