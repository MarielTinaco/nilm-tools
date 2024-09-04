
import numpy as np
import pandas as pd

class PowerSeriesToArrayFormatter:

        def format(self, data : pd.DataFrame, *args, **kwargs) -> np.ndarray:
                lens = [len(r) for r in data]
                min_lens = min(lens)
                p = [np.array(r)[:min_lens] for r in data]
                data = np.vstack(p)
                data[np.isnan(data)] = 0
                return data.T

        def __call__(self, data, *args, **kwargs):
                return self.format(data, *args, **kwargs)
