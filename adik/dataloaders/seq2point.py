
import numpy as np
from typing import Union, Iterable
from collections.abc import Sequence

class MultitargetQuantileRegressionSeq2PointDataLoader(Sequence):
	
	def __init__(self,
	      		 data : Union[np.ndarray, Iterable],
				 labels : Union[np.ndarray, Iterable],
				 seq_len : int,
				 indexer : object = None):

		self.data = data
		self.labels = labels
		self.seq_len = int(seq_len)

		self.indices = np.arange(self.data.shape[0]) if indexer is None else indexer(self.data.shape[0])

	def __get_sample(self, index):
		indices = self.indices[index : index + self.seq_len]
		inds_inputs = sorted(indices[:self.seq_len])
		inds_labels = sorted(indices[self.seq_len-1:self.seq_len])

		states = self.labels[0]
		power = self.labels[1]

		return self.data[inds_inputs], (states[inds_labels], power[inds_labels])

	def __len__(self):
		return (self.data.shape[0] - self.seq_len)

	def __getitem__(self, index):
		inputs, targets = self.__get_sample(index)
		state = targets[0]
		power = targets[1]
		return inputs, (state, power)


if __name__ == "__main__":

	from pathlib import Path

	PROFILE_PATH = Path.cwd() / "profiles" / "unetnilm_ukdale_20240730_143151"

	x = np.load(PROFILE_PATH / "training" / "noise_inputs.npy")
	y = np.load(PROFILE_PATH / "training" / "states.npy")
	z = np.load(PROFILE_PATH / "training" / "targets.npy")

	try:
		import torch

		class Seq2PointTorchDataset(torch.utils.data.Dataset):
			
			def __init__(self, input, labels, seq_len, indexer, sequence_strategy):
				self.backend = sequence_strategy(input, labels, seq_len, indexer)

			def __len__(self):
				return len(self.backend)

			def __getitem__(self, index):
				inputs, targets = self.backend[index]
				state = targets[0]
				power = targets[1]
				return torch.tensor(inputs).unsqueeze(-1).permute(1, 0).float(), \
					(torch.tensor(state).long().squeeze(), torch.tensor(power).float().squeeze())

		seq2point_torch = Seq2PointTorchDataset(x, (y, z), seq_len=100, indexer=np.arange, 
										sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader)

		print(len(seq2point_torch))
		print(seq2point_torch[0])
	
	except:
		print("torch not installed. can't run example")
	
	tf_installed = None
	try:
		import tensorflow as tf
		tf_installed = True
	except:
		print("tensorflow not installed")
		tf_installed = False
	
	if tf_installed:
		class Seq2PointTfDataset(tf.keras.utils.Sequence):

			class_dict = {"fridge" : 0, "washer dryer" : 1, "kettle" : 2, "dish washer" : 3, "microwave" : 4}

			def __init__(self, input, labels, seq_len, indexer, sequence_strategy, batch_size=256):
				self.batch_size = batch_size
				self.backend = sequence_strategy(input, labels, seq_len, indexer)

			def __len__(self):
				return len(self.backend) // self.batch_size

			def __getitem__(self, index):
				batch = [self.backend[idx] for idx in range(index * self.batch_size,(index + 1) * self.batch_size)]
				batch_data, batch_label = zip(*batch)

				batch_data = np.array(batch_data)
				batch_label = np.array(batch_label)

				batch_label = tf.squeeze(batch_label, axis=2)

				return batch_data, tuple([batch_label[:,0,:], batch_label[:,1,:]])

		seq2point_tf = Seq2PointTfDataset(x, (y, z), seq_len=100, indexer=np.arange, 
											sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader)
		print(len(seq2point_tf))
		print(seq2point_tf[0])