


import sys, os
from typing import Any
import numpy as np
import tensorflow as tf
import logging

from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import minmax_scale

import tensorflow as tf

from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot

from adik.dataloaders.seq2point import MultitargetQuantileRegressionSeq2PointDataLoader

from ..data_loader.seq2point_nilm import NILMSeq2PointDataset
from ..models import qat_simple_seq2point
from .. import parse_cmd
from .qat_training_seq2point import MultiActivationLoss, QuantileLoss, BaseAccuracy

from adinilm.utils.paths_manager import PROFILES_DIR, LOG_DIR, DATA_DIR


def convert_tflite_model(model):
	"""Convert the save TF model to tflite model, then save it as .tflite flatbuffer format

	"""
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()
	return tflite_model

def convert_quantized_tflite_model(model, x_train):
	"""Convert the save TF model to tflite model, then save it as .tflite flatbuffer format

	"""
	def representative_dataset_gen(num_samples=100):
		dset = iter(x_train)
		first_pop = next(dset)

		for idx in range(num_samples):
			data = first_pop[0][idx,:]
			data = np.reshape(data, (1, 100, 1))
			data = tf.cast(data, tf.float32)
			yield [data]

	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter.inference_input_type = tf.int8
	converter.inference_output_type = tf.int8
	converter.representative_dataset = representative_dataset_gen
	converter._experimental_disable_per_channel_quuantization_for_dense_layers=True
	tflite_model = converter.convert()
	return tflite_model


def run_main():
	args = parse_cmd.get_parser().parse_args()

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

	train_data = NILMSeq2PointDataset(PROFILE_PATH / "train", seq_len=SEQ_LEN,
				batch_size=BATCH_SIZE,
				sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader,
				input_transform = input_transform,
				label_transform = [lambda x: x, label1_transform])
	
	with tf.keras.utils.custom_object_scope({'MultiActivationLoss':MultiActivationLoss, 'QuantileLoss':QuantileLoss, 'BaseAccuracy':BaseAccuracy}):
		with tfmot.quantization.keras.quantize_scope():                                 
			model = tf.keras.models.load_model(CHECKPOINT)
			# ret = model.evaluate(data, batch_size=BATCH_SIZE)

			quantized_tflite_model_latest = convert_quantized_tflite_model(model, train_data)
			with open(CHECKPOINT.parent.parent / "tflite_micro_model_test.tflite", "wb") as f:
				f.write(quantized_tflite_model_latest)

	return None