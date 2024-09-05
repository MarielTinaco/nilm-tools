
import os
import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import minmax_scale

from adinilm.enumerations import *
from adinilm.objs import DatasetProfile, ProfileHandler, NormalizationHandler
from adinilm.io.profile import GenericProfileHandler

from adik.preprocessing.binarization import BinaryComparator
from adinilm.data_processing.formatter import PowerSeriesToArrayFormatter
from adinilm.filter.quantile_filter import QuantileFilterApplier, quantile_filter
from adinilm.augmentation.quantile_noiser import NoisedInput

from adinilm.utils.paths_manager import DATA_DIR, PROFILES_DIR


FILE_PATH = Path(__file__)


def run_main():

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

        for raw_file in os.listdir(raw_source_path):
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
                data_i_den = minmax_scale(data_i_den, feature_range=(-128, 127))
                data_i_n = noiser(data_i)
                data_i_n = minmax_scale(data_i_n, feature_range=(-128, 127))
                data = filt(data)
                data_p = np.apply_along_axis(norm, 0, data)
                data = data.T
                data_s = binarizer(data)
                data_s = data_s.T

                prof_handler.write(data_i_den, data_i_n, data_p, data_s, profile=prof, subdir=split_key)


if __name__ == "__main__":

        run_main()