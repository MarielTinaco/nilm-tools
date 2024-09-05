from pathlib import WindowsPath, Path
from adinilm.io.profile import *

def test_dataset_profiles_write_unetnilm():
        import shutil

        temp_dir = 'tests/temp_profile'
        handler = "unetnilm"

        Path(temp_dir).mkdir(exist_ok=True)

        prof = DatasetProfile(parent_dir=temp_dir,
                              handler=handler,
                              dataset_name="ukdale", 
                              metadata={},
                              mode="w")
        
        shutil.rmtree(temp_dir)

def test_dataset_profiles_write_and_read_unetnilm():

        import shutil

        temp_dir = 'tests/temp_profile'
        dataset_name = "ukdale"
        metadata = {}
        handler = "unetnilm"

        Path(temp_dir).mkdir(exist_ok=True)

        prof = DatasetProfile(parent_dir=temp_dir, 
                              handler=handler,
                              dataset_name=dataset_name, 
                              metadata=metadata, 
                              mode="w")
        
        full_path = prof.full_path

        new_dataset_profile = DatasetProfile.extract(full_path)

        assert new_dataset_profile.full_path == full_path
        assert new_dataset_profile.mode == "r"
        assert new_dataset_profile.metadata == metadata
        assert new_dataset_profile.dataset_name == dataset_name
        assert new_dataset_profile.parent_dir == WindowsPath(temp_dir)

        shutil.rmtree(temp_dir)

def test_dataset_profile_handler_unetnilm():

        import shutil
        import numpy as np
        from dataclasses import fields

        temp_dir = 'tests/temp_profile'
        handler = "unetnilm"
        data0 = np.zeros(1000)
        data1 = np.zeros(1000)
        data2 = np.zeros(1000)
        data3 = np.zeros(1000)

        Path(temp_dir).mkdir(exist_ok=True)

        prof = DatasetProfile(parent_dir=temp_dir,
                              handler=handler,
                              dataset_name="ukdale", 
                              metadata={},
                              mode="w")
        
        write_ret = ProfileHandler.write(data0, data1, data2, data3,
                                   profile=prof, subdir="training")
        
        assert prof == write_ret

        read_ret = ProfileHandler.read(prof.full_path, subdir="training")

        assert fields(read_ret[0]) == fields(write_ret)
        assert np.array_equal(read_ret[1], data0)
        assert np.array_equal(read_ret[2], data1)
        assert np.array_equal(read_ret[3], data2)
        assert np.array_equal(read_ret[4], data3)

        shutil.rmtree(temp_dir)
