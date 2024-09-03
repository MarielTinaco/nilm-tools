
import sys, os
import numpy as np
import tensorflow as tf
from pathlib import Path

class NILMSeq2PointDataset(tf.keras.utils.Sequence):

        def __init__(self, source, seq_len, sequence_strategy, batch_size=256, denoise=True, **kwargs):
                super().__init__(**kwargs)

                source = Path(source)

                x = np.load(source / "denoise_inputs.npy" if denoise else "noise_inputs.npy")
                y = np.load(source / "states.npy")
                z = np.load(source / "targets.npy")

                inputs = x
                labels = (y, z)

                indices = np.arange(x.shape[0])

                self.batch_size = batch_size
                self.backend = sequence_strategy(inputs, labels, seq_len)

        def __len__(self):
                return len(self.backend) // self.batch_size

        def __getitem__(self, index):
                batch = [self.backend[idx] for idx in range(index * self.batch_size,(index + 1) * self.batch_size)]
                batch_data, batch_label = zip(*batch)

                batch_data = np.array(batch_data)
                batch_label = np.array(batch_label)

                batch_label = tf.squeeze(batch_label, axis=2)

                return batch_data, tuple([batch_label[:,0,:], batch_label[:,1,:]])