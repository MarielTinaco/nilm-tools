


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

from adinilm.utils.paths_manager import PROFILES_DIR, LOG_DIR, DATA_DIR


def run_main():
	args = parse_cmd.get_parser().parse_args()

	SEQ_LEN = args.sequence_length
	PROFILE_PATH = Path(args.dataset) if args.dataset is not None else DATA_DIR / "NILMTK" / "processed"
	CHECKPOINT = Path(args.checkpoint)
	BATCH_SIZE = args.batch_size
	
	logdir_ = CHECKPOINT
	best_checkpoint_path = logdir_ / "weights" / "best.keras"

	data = NILMSeq2PointDataset(PROFILE_PATH / "test", seq_len=SEQ_LEN,
				batch_size=BATCH_SIZE,
				sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader)

	model = tf.keras.models.load_model(str(best_checkpoint_path.resolve()))

	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	tflite_model_best = converter.convert()
	with open(logdir_ / "model_best.tflite", "wb") as f:
		f.write(tflite_model_best)
	
	print("Quantized latest model in Mb:", os.path.getsize(logdir_ / "model_best.tflite") / float(2**20))

	return CHECKPOINT