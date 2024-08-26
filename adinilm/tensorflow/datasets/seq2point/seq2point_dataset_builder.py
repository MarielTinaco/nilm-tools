"""seq2point dataset."""

import sys, os
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from typing import Tuple

sys.path.append("../../../../")

from adinilm.utils.paths_manager import DATA_DIR
from adik.sampling.sampling import WindowSampler


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for seq2point dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(seq2point): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'input': tfds.features.Tensor(shape=(100,), dtype=np.float32),
            'label': tfds.features.Sequence(
              {
                'state' : tfds.features.Tensor(shape=(5,), dtype=np.int32),
                'rms'   : tfds.features.Tensor(shape=(5,), dtype=np.float32)
              }
            )}),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(seq2point): Downloads the data and defines the splits
    path = DATA_DIR / "NILMTK" / "processed"

    # TODO(seq2point): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        tfds.Split.TRAIN: self._generate_examples(path / 'train'),
        tfds.Split.TEST: self._generate_examples(path / 'test'),
        tfds.Split.VALIDATION: self._generate_examples(path / 'val'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    seq_len = 100
    stride = 1
    data = np.load(path / "denoise_inputs.npy")
    labels = np.load(path / "targets.npy"), np.load(path / "states.npy")

    output_stride = lambda x: np.arange(seq_len-1, labels[0].shape[0], stride)
    input_sampler = WindowSampler(data=data, length=seq_len, axis=0, stride=stride)
    rms_sampler = WindowSampler(data=labels[0], length=1, axis=0, stride= output_stride)
    states_sampler = WindowSampler(data=labels[1], length=1, axis=0, stride= output_stride)

    # TODO(seq2point): Yields (key, example) tuples from the dataset
    for i in range(len(rms_sampler)):
      yield str(i), {
          'input': input_sampler[i],
          'label': {
            'state' : states_sampler[i],
            'rms'   : rms_sampler[i],
          }
      }
