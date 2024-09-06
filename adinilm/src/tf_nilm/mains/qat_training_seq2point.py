

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

from ..callbacks.loggingcallback import PyLoggingCallback
from ..data_loader.seq2point_nilm import NILMSeq2PointDataset
from ..models import qat_simple_seq2point
from .. import parse_cmd

from adinilm.utils.paths_manager import PROFILES_DIR, LOG_DIR, DATA_DIR

FILE_PATH = Path(__file__)
quantize_model = tfmot.quantization.keras.quantize_model

class MultiActivationLoss(keras.losses.BinaryCrossentropy):

	def call(self, y_true, y_pred, **kwargs):
		y_true = tf.cast(y_true, dtype=np.uint8)
		true_state = tf.one_hot(y_true, depth=2)
		return super().call(true_state, y_pred, **kwargs)

class QuantileLoss(keras.losses.Loss):
	def __init__(self, quantiles=[0.0025,0.1, 0.5, 0.9, 0.975], *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.quantiles = quantiles

	def call(self, y_true, y_pred):
		# pred_rms = tf.reshape(y_pred, [-1, 5, 5])

		targets = tf.expand_dims(y_true, axis=1)
		targets = tf.repeat(targets, repeats=[5], axis=1)

		quantiles = tf.convert_to_tensor(self.quantiles, dtype=tf.float32)

		error = tf.transpose((targets - y_pred), perm=[0,2,1])
		loss = tf.math.maximum(quantiles*error, (quantiles-1)*error)
		return tf.reduce_mean(loss, axis=1)


class BaseAccuracy(keras.metrics.Accuracy):
    
        def update_state(self, y_true, y_pred, sample_weight=None):
                y_pred = tf.argmax(y_pred, axis=2)
                return super().update_state(y_true, y_pred, sample_weight)

def run_main():

	args = parse_cmd.get_parser().parse_args()

	SEQ_LEN = args.sequence_length
	PROFILE_PATH = Path(args.dataset) if args.dataset is not None else DATA_DIR / "NILMTK" / "processed"
	BATCH_SIZE = args.batch_size
	MODEL = args.model
	LEARNING_RATE = float(args.learning_rate)
	WEIGHT_DECAY = float(args.weight_decay)

	logdirname = datetime.now().strftime("%Y%m%d-%H%M%S")
	logdir = LOG_DIR / "tf_nilm"
	logdir.mkdir(exist_ok=True)

	logdir_ = logdir / logdirname
	logdir_.mkdir(exist_ok=True)
	ret = logdir_.resolve()
	weights = logdir_ / "weights"
	weights.mkdir(exist_ok=True)

	logfile = logdir_ / "train.log"

	train_data = NILMSeq2PointDataset(PROFILE_PATH / "train", seq_len=SEQ_LEN,
				batch_size=BATCH_SIZE,
				sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader)

	val_data = NILMSeq2PointDataset(PROFILE_PATH / "test", seq_len=SEQ_LEN,
				batch_size=BATCH_SIZE,
				sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader)

	tmp = FILE_PATH.parent.parent / "tmp"
	tmp.mkdir(exist_ok=True)

	model = qat_simple_seq2point.create_resnet_model(SEQ_LEN)
	# model = quantize_model(model)
	model = tfmot.quantization.keras.quantize_annotate_model(model)
	dot_img_file = tmp / 'resnet_model.png'
	keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
	optimizer = keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

	model.compile(optimizer=optimizer,
			loss={'quantize_annotate_53' : MultiActivationLoss(from_logits=False), 'quantize_annotate_54' : QuantileLoss()},
			metrics={'quantize_annotate_53' : BaseAccuracy()})

	model.summary()

	best_checkpoint_path = weights / "best.keras"
	last_checkpoint_path = weights / "checkpoint.keras"
	logger_callback = PyLoggingCallback(filename=logfile, encoding='utf-8', level=logging.INFO)
	
	logging.info(f"Profile used: {PROFILE_PATH.resolve()}")
	lrscheduler_callback = keras.callbacks.ReduceLROnPlateau(monitor = "val_quantize_annotate_53_accuracy",
								    mode = "max")
	tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir_)
	best_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=str(best_checkpoint_path),
									monitor='val_quantize_annotate_53_accuracy',
									mode='max',
									save_best_only=True,
									save_weights_only=False,
									initial_value_threshold=0.41,
									verbose=1)
	last_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=str(last_checkpoint_path),
									save_weights_only=False,
									verbose=1)
	early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_quantize_annotate_53_accuracy',
						min_delta=0.04,
						patience=5,
						verbose=0,
						mode='max',
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

	model.fit(train_data,
	   	  epochs=args.epochs,
		  callbacks=cb_list,
		  validation_data=val_data)

	model.export(str(logdir_ / "test"), "tf_saved_model")
	
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	tflite_model = converter.convert()
	with open(logdir_ / "model.tflite", "wb") as f:
		f.write(tflite_model)
	
	print("Quantized model in Mb:", os.path.getsize(logdir_ / "model.tflite") / float(2**20))

	return ret