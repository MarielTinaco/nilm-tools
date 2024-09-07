


import sys, os
from typing import Any
import numpy as np
import tensorflow as tf
import logging

from pathlib import Path
from datetime import datetime

import tensorflow as tf

from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot

from adik.dataloaders.seq2point import MultitargetQuantileRegressionSeq2PointDataLoader

from ..data_loader.seq2point_nilm import NILMSeq2PointDataset
from ..models import qat_simple_seq2point
from .. import parse_cmd
from .qat_training_seq2point import MultiActivationLoss, QuantileLoss, BaseAccuracy

from adinilm.utils.paths_manager import PROFILES_DIR, LOG_DIR, DATA_DIR


def run_main():
	args = parse_cmd.get_parser().parse_args()

	SEQ_LEN = args.sequence_length
	PROFILE_PATH = Path(args.dataset) if args.dataset is not None else DATA_DIR / "NILMTK" / "processed"
	CHECKPOINT = Path(args.checkpoint)
	BATCH_SIZE = args.batch_size
	
	data = NILMSeq2PointDataset(PROFILE_PATH / "val", seq_len=SEQ_LEN,
				batch_size=BATCH_SIZE,
				sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader)
	
	with tf.keras.utils.custom_object_scope({'MultiActivationLoss':MultiActivationLoss, 'QuantileLoss':QuantileLoss, 'BaseAccuracy':BaseAccuracy}):
		with tfmot.quantization.keras.quantize_scope():                                 
			model = tf.keras.models.load_model(CHECKPOINT)
		ret = model.evaluate(data, batch_size=BATCH_SIZE)

	return ret