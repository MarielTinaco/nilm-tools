
import numpy as np
import adinilm.functions as F


def quantile_filter(data, sequence_length, p=50):
    new_mains = F.scan_sequences(data=data, mode="odd_window", seq_len=sequence_length)
    new_mains = np.percentile(new_mains, p, axis=1, interpolation="nearest")
    return new_mains

class QuantileFilterApplier:

        def __init__(self, window, p=50):
                self.window = window
                self.p = p
        
        def filter(self, data, *args, **kwargs):
                data_c = data.copy()
                for i in range(len(self.window)):
                        data_c[:,i] = quantile_filter(data[:,i], self.window[i], p=self.p)
                return data_c

        def __call__(self, data, *args, **kwargs):
                return self.filter(data, *args, **kwargs)


if __name__ == "__main__":

        window = [50, 50, 50, 10, 50]
        sample_data = np.random.randint(0, 2000, size=(5, 100))

        filt = QuantileFilterApplier(window, p=50)

        print(filt(sample_data))
