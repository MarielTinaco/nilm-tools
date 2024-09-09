
import sys, os
import numpy as np
import tensorflow as tf
import logging
import json
import pickle
import functools

from typing import Any
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from adik.dataloaders.seq2point import MultitargetQuantileRegressionSeq2PointDataLoader
from adinilm.enumerations import *
from adinilm.objs import DatasetProfile, ProfileHandler, NormalizationHandler
from adinilm.io.profile import GenericProfileHandler

from adik.preprocessing.binarization import BinaryComparator
from adinilm.data_processing.formatter import PowerSeriesToArrayFormatter
from adinilm.filter.quantile_filter import QuantileFilterApplier, quantile_filter
from adinilm.augmentation.quantile_noiser import NoisedInput

from ..callbacks.loggingcallback import PyLoggingCallback
from ..data_loader.seq2point_nilm import NILMSeq2PointDataset
from ..models import simple_seq2point
from ..metrics.baseaccuracy import BaseAccuracy
from ..losses.quantileloss import QuantileLoss
from ..losses.multiactivationloss import MultiActivationLoss 
from .. import parse_cmd

from adinilm.utils.paths_manager import PROFILES_DIR, LOG_DIR, DATA_DIR

FILE_PATH = Path(__file__)

def check_processed_files_exist():
	def check_npy_files(path):
		print(f"checking contents of {path.resolve()}")
		ldir = set(os.listdir(path))
		noised_input_check = "noise_inputs.npy" in ldir
		denoised_input_check = "denoise_inputs.npy" in ldir
		states_check = "states.npy" in ldir
		targets_check = "targets.npy" in ldir

		return all([noised_input_check, denoised_input_check, states_check, targets_check])

	processed_dest_path = DATA_DIR / "NILMTK" / "processed"

	print(f"Checking processed folder {processed_dest_path.resolve()}")
	for dir in ["test", "train", "val"]:
		proc_dir = processed_dest_path / dir
		proc_dir.mkdir(exist_ok=True)
		check = check_npy_files(proc_dir)

		if not check:
			print(f"Missing files in {dir}")
			return False

	return True
		

def dataset_pipeline():
	raw_source_path = DATA_DIR / "NILMTK" / "raw"
	processed_dest_path = DATA_DIR / "NILMTK" / "processed"
	processed_dest_path.mkdir(exist_ok=True)

	config_path = FILE_PATH.parent.parent / "configs" / "nilmtk_extract.json"

	with open(config_path, "r") as config_file:
		config = json.load(config_file)

	prof = DatasetProfile(parent_dir=PROFILES_DIR,
			handler=NILMProjects.UNETNILM.value,
			dataset_name="ukdale",
			metadata=config,
			mode="w")
	prof.full_path = processed_dest_path
	prof_handler = GenericProfileHandler(PROFILES_DIR)

	for raw_file in tqdm(os.listdir(raw_source_path)):
		full_raw_path = raw_source_path / raw_file
		split_key = raw_file.split(".")[0]

		window = [i["window"] for i in config[split_key]["appliances"]]

		with open(full_raw_path, 'rb') as raw_pickle:
			raw_data = pickle.load(raw_pickle)

		power_series = raw_data.loc[raw_data.index != "site meter","power_series"]
		on_power_threshold = raw_data.loc[raw_data.index != "site meter","on_power_threshold"]

		formatter = PowerSeriesToArrayFormatter()
		filt = QuantileFilterApplier(window)
		mixer = lambda data : data.sum(axis=1)
		noiser = NoisedInput()
		binarizer = BinaryComparator(np.array(on_power_threshold))
		norm = lambda data : minmax_scale(data, feature_range=(0, 1))

		data = formatter(power_series)
		data_i = mixer(data)
		data_i_den = quantile_filter(data_i, 10, p=50)
		data_i_n = noiser(data_i)
		data = filt(data)
		data_p = data
		data = data.T
		data_s = binarizer(data)
		data_s = data_s.T

		prof_handler.write(data_i_den, data_i_n, data_p, data_s, profile=prof, subdir=split_key)

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

	if not check_processed_files_exist():
		dataset_pipeline()

	def input_transform(data):
		data = minmax_scale(data, feature_range=(-128, 127))
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

	val_data = NILMSeq2PointDataset(PROFILE_PATH / "test", seq_len=SEQ_LEN,
				batch_size=BATCH_SIZE,
				sequence_strategy= MultitargetQuantileRegressionSeq2PointDataLoader,
				input_transform = input_transform,
				label_transform = [lambda x: x, label1_transform])

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
			loss={'y1_output' : MultiActivationLoss(from_logits=False), 'y2_output' : QuantileLoss()},
			metrics={'y1_output' : BaseAccuracy()})

	model.summary()

	best_checkpoint_path = weights / "best.keras"
	last_checkpoint_path = weights / "checkpoint.keras"
	logger_callback = PyLoggingCallback(filename=logfile, encoding='utf-8', level=logging.INFO)
	
	logging.info(f"Profile used: {PROFILE_PATH.resolve()}")
	lrscheduler_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_y1_output_accuracy",
								    mode = "max")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir_)
	best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=best_checkpoint_path,
									monitor='val_y1_output_accuracy',
									mode='max',
									save_best_only=True,
									save_weights_only=False,
									initial_value_threshold=0.41,
									verbose=1)
	last_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=last_checkpoint_path,
									save_weights_only=False,
									verbose=1)
	early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_y1_output_accuracy',
						min_delta=0.04,
						patience=5,
						verbose=0,
						mode='max',
						baseline=None,
						restore_best_weights=False,
						start_from_epoch=40
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


	return ret