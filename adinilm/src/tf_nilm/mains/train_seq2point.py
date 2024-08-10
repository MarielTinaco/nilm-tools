
import sys, os
import numpy as np
import tensorflow as tf
import logging

from pathlib import Path
from datetime import datetime

from adik.dataloaders.seq2point import MultitargetQuantileRegressionSeq2PointDataLoader

from ..callbacks.loggingcallback import PyLoggingCallback
from ..data_loader.seq2point_nilm import NILMSeq2PointDataset
from ..models import simple_seq2point
from .. import parse_cmd
from ..losses.quantileloss import QuantileLoss 
from ..losses.multiactivationloss import MultiActivationLoss 

from adinilm.utils.paths_manager import PROFILES_DIR, LOG_DIR

def run_main():

	args = parse_cmd.get_parser().parse_args()

	SEQ_LEN = args.sequence_length
	PROFILE_PATH = args.dataset_profile if args.dataset_profile is not None else PROFILES_DIR / "unetnilm_ukdale_20240321_155419"
	BATCH_SIZE = args.batch_size

	logdirname = datetime.now().strftime("%Y%m%d-%H%M%S")
	logdir = LOG_DIR / "tf_nilm"
	logdir.mkdir(exist_ok=True)

	logdir_ = logdir / logdirname
	logdir_.mkdir(exist_ok=True)

	weights = logdir_ / "weights"
	weights.mkdir(exist_ok=True)

	logfile = logdir_ / "train.log"

	x = np.load(PROFILE_PATH / "training" / "noise_inputs.npy")
	y = np.load(PROFILE_PATH / "training" / "states.npy")
	z = np.load(PROFILE_PATH / "training" / "targets.npy")

	data = NILMSeq2PointDataset(x, (y, z), seq_len=SEQ_LEN, indexer=np.arange,
						             batch_size=BATCH_SIZE,
							     sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader)
	model = simple_seq2point.create_model(SEQ_LEN)

	dot_img_file = '/tmp/resnet_model.png'
	if not Path('/tmp').exists():
		Path.mkdir(Path('/tmp'))

	tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

	model.compile(optimizer=tf.keras.optimizers.Adam(),
			loss={'y1_output' : MultiActivationLoss, 'y2_output' : QuantileLoss})

	model.summary()

	best_checkpoint_path = weights / "best-epoch-{epoch}.weights.h5"
	last_checkpoint_path = weights / "checkpoint.weights.h5"
	logger_callback = PyLoggingCallback(filename=logfile, encoding='utf-8', level=logging.INFO)
	lrscheduler_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss",
								    mode = "min")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir_)
	best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=best_checkpoint_path,
									monitor='val_loss',
									mode='min',
									save_weights_only=True,
									initial_value_threshold=0.41,
									verbose=1)
	last_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=last_checkpoint_path,
									save_weights_only=True,
									verbose=1)
	cb_list = [tensorboard_callback,
	    	   lrscheduler_callback,
		   best_checkpoint_callback,
		   last_checkpoint_callback,
		   logger_callback]

	model.fit(data, epochs=args.epochs, callbacks=cb_list)
	model.export(str(logdir_ / "test"), "tf_saved_model")
	
	converter = tf.lite.TFLiteConverter.from_saved_model(str(logdir_ / "test"))
	tflite_model = converter.convert()
	with open(logdir_ / "model.tflite", "wb") as f:
		f.write(tflite_model)

	ret = logdir_.resolve()

	return ret