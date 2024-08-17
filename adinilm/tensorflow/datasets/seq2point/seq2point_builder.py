
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

from adinilm.objs import DatasetProfile, ProfileHandler


class Seq2PointBuilder(tfds.dataset_builders.TfDataBuilder):

        def __init__(self, profile_path, split = 0.1):

                prof, x1, x2, y, z = ProfileHandler.read(profile_path, subdir="training")
                index_split = int(split * len(x1))

                x1_train = x1[:index_split]
                x1_test = x1[index_split:]

                x2_train = x2[:index_split]
                x2_test = x2[index_split:]

                y_train = y[:index_split]
                y_test = y[index_split:]

                z_train = z[:index_split]
                z_test = z[index_split:]

                super().__init__(
                        name= "seq2point",
                        version= "1.0.0",
                        split_datasets = {
                                "train": (x1_train, x2_train, y_train, z_train),
                                "test":  (x1_test, x2_test, y_test, z_test)
                        },
                        features = tfds.features.FeaturesDict({
                                "input": tfds.features.Tensor(shape=(100, 1), dtype=np.float64),
                                "states": tfds.features.Tensor(shape=(1, 5), dtype=np.float64),
                                "power": tfds.features.Tensor(shape=(1, 5), dtype=np.float64),
                        }),
                        config='nilm',
                        description='Seq2Point dataset',
                        release_notes={
                                "1.0.0": "Initial release"
                        }
                )
                
                
