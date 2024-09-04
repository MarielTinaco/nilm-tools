import numpy as np

class BinaryComparator:

        def __init__(self, threshold : np.ndarray):
                self.threshold = threshold
        
        def forward(self, data : np.ndarray):
                assert data.shape[0] == self.threshold.shape[0], \
                        f"Shape mismatch: {data.shape[0]}, self.threshold.shape[0]"

                threshold = np.empty(shape=data.shape).T
                threshold[:] = self.threshold
                threshold = threshold.T

                return np.where(data >= threshold, 1, 0).astype(int)

        def __call__(self, data : np.ndarray, *args, **kwargs):
                return self.forward(data)



if __name__ == "__main__":

        sample_data = np.random.sample((4, 5, 3))

        comp = BinaryComparator(threshold=np.array([0.5, 0.5, 0.5]))

        output = (comp(sample_data))