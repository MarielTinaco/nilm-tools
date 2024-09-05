import pytest
import random
import nilmtk

import adinilm.utils.paths_manager as pathsman
from adinilm.augmentation.activation_extension import *

def create_test_nilmtk_elecmeter_data():
        BUILDING = 1
        TIME_START = "2015-01-01"
        TIME_END = "2015-01-15"
        APPLIANCE = "fridge"

        dataset = nilmtk.DataSet(pathsman.UKDALE_H5_PATH)
        dataset.set_window(start=TIME_START, end=TIME_END)
        elecmeter = dataset.buildings[BUILDING].elec[APPLIANCE]

        return elecmeter

def test_elecmeter_activation_extension():

        TEST_NUM_SAMPLES = 100

        elecmeter = create_test_nilmtk_elecmeter_data()

        assert type(elecmeter) == nilmtk.ElecMeter

        strat = ElecmeterActivationAppender(data=elecmeter)
        ctx = ActivationExtensionContext(strategy=strat)

        assert hasattr(ctx.strategy, "extend") == True

        extended = ctx.extend(num_samples=TEST_NUM_SAMPLES)

        assert len(extended) == len(elecmeter.power_series_all_data()) + TEST_NUM_SAMPLES


@pytest.mark.parametrize(
        "dummy_data, mode, full_num_samples, extras",
        [(create_test_nilmtk_elecmeter_data(), None, 200000, {"interval" : 10}),
         (create_test_nilmtk_elecmeter_data(), "appender", 200000, {"interval" : 10}),
         (create_test_nilmtk_elecmeter_data(), "randomizer", 200000, {"interval" : 10}),
         (create_test_nilmtk_elecmeter_data(), "randomizer", 200000, {"interval" : 1000}),
         (create_test_nilmtk_elecmeter_data(), "rightpadder", 200000, {"padding_mode" : 0}),
         (create_test_nilmtk_elecmeter_data(), "rightpadder", 200000, {"padding_mode" : lambda : random.randint(1, 4)}),
         (create_test_nilmtk_elecmeter_data(), None, 200000, {})]
)
def test_activation_extension_interface_function(dummy_data, mode, full_num_samples, extras):

        extended = extend_activations(dummy_data,
                                      num_full_samples=full_num_samples,
                                      mode= mode,
                                      **extras)

        assert len(extended) == full_num_samples