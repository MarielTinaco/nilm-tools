
import sys, os
import numpy as np
import tensorflow as tf

class NILMSeq2PointDataset(tf.keras.utils.Sequence):

        def __init__(self, input, labels, seq_len, indices, sequence_strategy, batch_size=256):
                self.batch_size = batch_size
                self.backend = sequence_strategy(input, labels, seq_len, indices)

        def __len__(self):
                return len(self.backend) // self.batch_size

        def __getitem__(self, index):
                batch = [self.backend[idx] for idx in range(index * self.batch_size,(index + 1) * self.batch_size)]
                batch_data, batch_label = zip(*batch)

                batch_data = np.array(batch_data)
                batch_label = np.array(batch_label)

                batch_label = tf.squeeze(batch_label, axis=2)

                return batch_data, tuple([batch_label[:,0,:], batch_label[:,1,:]])