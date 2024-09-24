import sys, os
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import minmax_scale
from pathlib import Path

from adik.dataloaders.seq2point import MultitargetQuantileRegressionSeq2PointDataLoader

from ..data_loader.seq2point_nilm import NILMSeq2PointDataset
from ..models import simple_seq2point
from .. import parse_cmd
from ..losses.quantileloss import QuantileLoss 
from ..losses.multiactivationloss import MultiActivationLoss 
from adinilm.utils.paths_manager import PROFILES_DIR, LOG_DIR, DATA_DIR

def run_main(args):

	SEQ_LEN = args.sequence_length
	PROFILE_PATH = Path(args.dataset) if args.dataset is not None else DATA_DIR / "NILMTK" / "processed"
	CHECKPOINT = Path(args.checkpoint)
	BATCH_SIZE = args.batch_size

	def input_transform(data):
		data = minmax_scale(data, feature_range=(0, 1))
		return data

	def label1_transform(data):
		norm = lambda data : minmax_scale(data, feature_range=(0, 1))
		data = np.apply_along_axis(norm, 0, data)
		return data

	data = NILMSeq2PointDataset(PROFILE_PATH / "val", seq_len=SEQ_LEN,
				batch_size=BATCH_SIZE,
				sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader,
				input_transform = input_transform,
				label_transform = [lambda x: x, label1_transform])

	model = tf.keras.models.load_model(CHECKPOINT)

	ret = model.predict(data, batch_size=BATCH_SIZE)

	return ret
