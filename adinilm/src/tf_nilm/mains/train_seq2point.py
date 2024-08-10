
import sys, os
import numpy as np
import tensorflow as tf

from pathlib import Path
from datetime import datetime

from adik.dataloaders.seq2point import MultitargetQuantileRegressionSeq2PointDataLoader

from adinilm.src.tf_nilm.data_loader.seq2point_nilm import MultitargetQuantileRegressionSeq2PointDataset
from adinilm.src.tf_nilm.models import simple_seq2point
from adinilm.src.tf_nilm import parse_cmd
from adinilm.src.tf_nilm.losses.quantileloss import QuantileLoss 
from adinilm.src.tf_nilm.losses.multiactivationloss import MultiActivationLoss 
from adinilm.utils.paths_manager import PROFILES_DIR, LOG_DIR

def run_main():

	args = parse_cmd.get_parser().parse_args()

	SEQ_LEN = args.sequence_length
	PROFILE_PATH = PROFILES_DIR / "unetnilm_ukdale_20240321_155419"

	x = np.load(PROFILE_PATH / "training" / "noise_inputs.npy")
	y = np.load(PROFILE_PATH / "training" / "states.npy")
	z = np.load(PROFILE_PATH / "training" / "targets.npy")

	data = MultitargetQuantileRegressionSeq2PointDataset(x, (y, z), seq_len=SEQ_LEN, indexer=np.arange,
							sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader)
	model = simple_seq2point.create_model(SEQ_LEN)

	dot_img_file = '/tmp/resnet_model.png'
	if not Path('/tmp').exists():
		Path.mkdir(Path('/tmp'))

	tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

	model.compile(optimizer=tf.keras.optimizers.Adam(),
			loss={'y1_output' : MultiActivationLoss, 'y2_output' : QuantileLoss})

	model.summary()

	logdirname = datetime.now().strftime("%Y%m%d-%H%M%S")
	logdir = LOG_DIR / "tf_nilm"
	logdir.mkdir(exist_ok=True)

	logdir_ = logdir / logdirname
	logdir_.mkdir(exist_ok=True)
	
	weights = logdir_ / "weights"
	weights.mkdir(exist_ok=True)

	checkpoint_path = weights / "cp.weights.h5" 
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir_)
	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
								 save_weights_only=True,
								 verbose=1)

	model.fit(data, epochs=args.epochs, callbacks=[tensorboard_callback, checkpoint_callback])

	ret = logdir_.resolve()

	return ret