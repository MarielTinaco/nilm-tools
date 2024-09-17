
import os
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from adinilm.enumerations import *
from adinilm.io.profile import GenericProfileHandler
from adinilm.objs import DatasetProfile

from adik.preprocessing.binarization import BinaryComparator
from adinilm.data_processing.formatter import PowerSeriesToArrayFormatter
from adinilm.filter.quantile_filter import QuantileFilterApplier, quantile_filter
from adinilm.augmentation.quantile_noiser import NoisedInput
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
