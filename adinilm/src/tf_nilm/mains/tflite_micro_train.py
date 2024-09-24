
"""Seq2Point training for NILM


"""

import os
import logging
import numpy as np
import tensorflow as tf

from enum import Enum
from typing import Union
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import minmax_scale

from ..models import simple_seq2point
from ..metrics.baseaccuracy import BaseAccuracy
from ..losses.quantileloss import QuantileLoss
from ..losses.multiactivationloss import MultiActivationLoss 
from ..callbacks.loggingcallback import PyLoggingCallback
from ..data_loader.seq2point_nilm import NILMSeq2PointDataset
from ..data_proc.pipeline import check_processed_files_exist, dataset_pipeline

from adik.dataloaders.seq2point import MultitargetQuantileRegressionSeq2PointDataLoader

from adinilm.utils.paths_manager import PROFILES_DIR, LOG_DIR, DATA_DIR

FILE_PATH = Path(__file__)

class Seq2PointModel(Enum):
	SIMPLE = "simple"
	RESNET = "resnet"

def create_model(modeltype : Union[str, Seq2PointModel], seq_len, optimizer):

	if Seq2PointModel(modeltype) == Seq2PointModel.RESNET:
		model = simple_seq2point.create_resnet_model(seq_len)
	else:
		model = simple_seq2point.create_model(seq_len)

	model.compile(optimizer=optimizer,
	       		  loss={'y1_output' : MultiActivationLoss(from_logits=False), 'y2_output' : QuantileLoss()},
				  metrics={'y1_output' : BaseAccuracy()})
	
	model.summary()
	return model


def train_model(train_data, val_data, epochs, modeltype, seq_len, optimizer, cb_list):

	model = create_model(modeltype, seq_len, optimizer)

	model.fit(train_data,
	   	  epochs=epochs,
		  callbacks=cb_list,
		  validation_data=val_data)

	return model


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
	# converter._experimental_disable_per_channel_quuantization_for_dense_layers=True
	tflite_model = converter.convert()
	return tflite_model


def prepare_trained_model(trained_model):
	"""Fix the input of the trained model for inference

	Args:
		trained_model (tf.keras.Model): the trained LSTM model

	Returns:
		run_model (tf.keras.Model): the trained model with fixed input tensor size for inference
	"""
	# TFLite converter requires fixed shape input to work, alternative: b/225231544
	fixed_input = tf.keras.layers.Input(shape=(100,1),
					batch_size=1,
					dtype=trained_model.inputs[0].dtype,
					name="fixed_input")
	fixed_output = trained_model(fixed_input)
	run_model = tf.keras.models.Model(fixed_input, fixed_output)
	return run_model


def save_tflite_model(tflite_model, save_dir, model_name):
	"""save the converted tflite model

	Args:
		tflite_model (binary): the converted model in serialized format.
		save_dir (str): the save directory
		model_name (str): model name to be saved
	"""
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	save_path = os.path.join(save_dir, model_name)
	with open(save_path, "wb") as f:
		f.write(tflite_model)
	logging.info("Tflite model saved to %s", save_dir)
	logging.info("TFlite model named: {} is {} Mb in size".format(str(save_path), os.path.getsize(save_path) / float(2**20)))

def run_main(args):

	SEQ_LEN = args.sequence_length
	PROFILE_PATH = Path(args.dataset) if args.dataset is not None else DATA_DIR / "NILMTK" / "processed"
	BATCH_SIZE = args.batch_size
	MODEL = args.model
	LEARNING_RATE = float(args.learning_rate)
	WEIGHT_DECAY = float(args.weight_decay)
	MONITOR = str(args.monitor)
	MONITOR_MODE = str(args.monitor_mode)
	EPOCHS = int(args.epochs)

	logdirname = datetime.now().strftime("%Y%m%d-%H%M%S")
	logdir = LOG_DIR / "tf_nilm"
	logdir.mkdir(exist_ok=True)

	logdirname = logdir / logdirname
	logdirname.mkdir(exist_ok=True)
	logdirname = logdirname.resolve()

	if not check_processed_files_exist():
		dataset_pipeline()

	def input_transform(data):
		data = minmax_scale(data, feature_range=(0, 1))
		return data

	def label1_transform(data):
		norm = lambda data : minmax_scale(data, feature_range=(-128, 127))
		data = np.apply_along_axis(norm, 0, data)
		return data

	train_data = NILMSeq2PointDataset(PROFILE_PATH / "train", seq_len=SEQ_LEN,
				batch_size=BATCH_SIZE,
				sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader,
				input_transform = input_transform,
				label_transform = [lambda x: x, label1_transform])

	val_data = NILMSeq2PointDataset(PROFILE_PATH / "test", seq_len=SEQ_LEN,
				batch_size=BATCH_SIZE,
				sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader,
				input_transform = input_transform,
				label_transform = [lambda x: x, label1_transform])

	tmp = FILE_PATH.parent.parent / "tmp"
	tmp.mkdir(exist_ok=True)

	weights = logdirname / "weights"
	weights.mkdir(exist_ok=True)

	optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

	best_checkpoint_path = weights / "best.keras"
	last_checkpoint_path = weights / "checkpoint.keras"
	logger_callback = PyLoggingCallback(filename=logdirname / 'train.log', encoding='utf-8', level=logging.INFO)
	
	# logging.info(f"Profile used: {PROFILE_PATH.resolve()}")
	lrscheduler_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor = MONITOR,
							mode = MONITOR_MODE)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdirname)
	best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(best_checkpoint_path),
									monitor=MONITOR,
									mode=MONITOR_MODE,
									save_best_only=True,
									save_weights_only=False,
									initial_value_threshold=1.5,
									verbose=1)
	last_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(last_checkpoint_path),
									save_weights_only=False,
									verbose=1)
	early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor=MONITOR,
						min_delta=0.04,
						patience=5,
						verbose=0,
						mode=MONITOR_MODE,
						baseline=None,
						restore_best_weights=False,
						start_from_epoch=15
						)

	cb_list = [tensorboard_callback,
	    	   lrscheduler_callback,
			   best_checkpoint_callback,
			   last_checkpoint_callback,
			   early_stop_callback,
			   logger_callback]

	model = train_model(train_data, val_data, EPOCHS, MODEL, SEQ_LEN, optimizer, cb_list)
	run_model = prepare_trained_model(model)

	run_model.save(weights / "trained.keras")
	logging.info("TF model saved to %s", weights / "trained.keras")

	# Convert and save the model to .tflite
	tflite_model = convert_tflite_model(run_model)
	save_tflite_model(tflite_model,
			weights,
			model_name="seq2point_nilm.tflite")

	quantized_tflite_model = convert_quantized_tflite_model(run_model, train_data)
	save_tflite_model(quantized_tflite_model,
			weights,
			model_name="seq2point_nilm_quant.tflite")

	model = tf.keras.models.load_model(best_checkpoint_path) 
	run_model = prepare_trained_model(model)

	# Convert and save the model to .tflite
	tflite_model = convert_tflite_model(run_model)
	save_tflite_model(tflite_model,
			weights,
			model_name="seq2point_nilm_best.tflite")

	quantized_tflite_model = convert_quantized_tflite_model(run_model, train_data)
	save_tflite_model(quantized_tflite_model,
			weights,
			model_name="seq2point_nilm_quant_best.tflite")

	return {
		'logs' : logdirname
	}

