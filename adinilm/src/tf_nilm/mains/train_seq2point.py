
import sys, os
from typing import Any
import numpy as np
import tensorflow as tf
import logging

from pathlib import Path
from datetime import datetime

from tensorflow.keras.optimizers import Adam

from adik.dataloaders.seq2point import MultitargetQuantileRegressionSeq2PointDataLoader

from ..callbacks.loggingcallback import PyLoggingCallback
from ..data_loader.seq2point_nilm import NILMSeq2PointDataset
from ..models import simple_seq2point
from ..losses.quantileloss import QuantileLoss 
from ..losses.multiactivationloss import MultiActivationLoss 
from .. import parse_cmd

from adinilm.utils.paths_manager import PROFILES_DIR, LOG_DIR

FILE_PATH = Path(__file__)

def run_main():

	args = parse_cmd.get_parser().parse_args()

	SEQ_LEN = args.sequence_length
	PROFILE_PATH = Path(args.dataset) if args.dataset is not None else PROFILES_DIR / "unetnilm_ukdale_20240321_155419"
	BATCH_SIZE = args.batch_size
	MODEL = args.model
	LEARNING_RATE = float(args.learning_rate)
	WEIGHT_DECAY = float(args.weight_decay)

	logdirname = datetime.now().strftime("%Y%m%d-%H%M%S")
	logdir = LOG_DIR / "tf_nilm"
	logdir.mkdir(exist_ok=True)

	logdir_ = logdir / logdirname
	logdir_.mkdir(exist_ok=True)

	weights = logdir_ / "weights"
	weights.mkdir(exist_ok=True)

	logfile = logdir_ / "train.log"

	x_train = np.load(PROFILE_PATH / "train" / "denoise_inputs.npy")
	y_train = np.load(PROFILE_PATH / "train" / "states.npy")
	z_train = np.load(PROFILE_PATH / "train" / "targets.npy")

	indices = np.arange(x_train.shape[0])

	train_data = NILMSeq2PointDataset(x_train, (y_train, z_train), seq_len=SEQ_LEN, indices=indices,
				batch_size=BATCH_SIZE,
				sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader)
	
	x_val = np.load(PROFILE_PATH / "val" / "denoise_inputs.npy")
	y_val = np.load(PROFILE_PATH / "val" / "states.npy")
	z_val = np.load(PROFILE_PATH / "val" / "targets.npy")

	indices = np.arange(x_val.shape[0])

	val_data = NILMSeq2PointDataset(x_val, (y_val, z_val), seq_len=SEQ_LEN, indices=indices,
				batch_size=BATCH_SIZE,
				sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader)

	tmp = FILE_PATH.parent.parent / "tmp"
	tmp.mkdir(exist_ok=True)

	if MODEL == "resnet":
		model = simple_seq2point.create_resnet_model(SEQ_LEN)
		dot_img_file = tmp / 'resnet_model.png'
	else:
		model = simple_seq2point.create_model(SEQ_LEN)
		dot_img_file = tmp / 'simplenet_model.png'

	tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

	optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

	model.compile(optimizer=optimizer,
			loss={'y1_output' : MultiActivationLoss(), 'y2_output' : QuantileLoss()})

	model.summary()

	best_checkpoint_path = weights / "best-epoch-{epoch}.keras"
	last_checkpoint_path = weights / "checkpoint.keras"
	logger_callback = PyLoggingCallback(filename=logfile, encoding='utf-8', level=logging.INFO)
	
	logging.info(f"Profile used: {PROFILE_PATH.resolve()}")
	lrscheduler_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss",
								    mode = "min")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir_)
	best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=best_checkpoint_path,
									monitor='val_loss',
									mode='min',
									save_weights_only=False,
									initial_value_threshold=0.41,
									verbose=1)
	last_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=last_checkpoint_path,
									save_weights_only=False,
									verbose=1)
	early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
						min_delta=0,
						patience=0,
						verbose=0,
						mode='auto',
						baseline=None,
						restore_best_weights=False,
						start_from_epoch=0
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
	
	converter = tf.lite.TFLiteConverter.from_saved_model(str(logdir_ / "test"))
	tflite_model = converter.convert()
	with open(logdir_ / "model.tflite", "wb") as f:
		f.write(tflite_model)

	ret = logdir_.resolve()

	return ret