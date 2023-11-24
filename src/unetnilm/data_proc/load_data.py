import sys
import pandas as pd
import numpy as np
from pathlib import Path
import nilmtk   
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

sys.path.append('../../')
from src.utils import paths_manager as pathsman
from utils.mappings import ukdale_appliance_data


# data_type = ("training")
save_path = ('data')

def binarization(data,threshold):
    """[summary]
    
    Arguments:
        data {[type]} -- [description]
        threshold {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    state = np.where(data>= threshold,1,0).astype(int)
    return state

def get_percentile(data,p=50):
    """[summary]
    
    Arguments:
        data {[type]} -- [description]
        quantile {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return np.percentile(data, p, axis=1, interpolation="nearest")

def generate_sequences(sequence_length, data):
    sequence_length = sequence_length - 1 if sequence_length% 2==0 else sequence_length
    units_to_pad = sequence_length // 2
    new_mains = np.pad(data, (units_to_pad,units_to_pad),'constant',constant_values=(0,0))
    new_mains = np.array([new_mains[i:i + sequence_length] for i in range(len(new_mains) - sequence_length+1)])
    return new_mains

def quantile_filter(sequence_length, data, p=50):
    new_mains = generate_sequences(sequence_length, data)
    new_mains = get_percentile(new_mains, p)
    return new_mains

def pre_proc_ukdale(data_type, window):
    targets = []
    states = [] 
    dataset = nilmtk.DataSet(pathsman.UKDALE_H5_PATH)
    dataset.set_window(*window)
    power_elec = dataset.buildings[1].elec
    
    washer_dryer_power = power_elec["washer dryer"].power_series_all_data()
    kettle_power = power_elec["kettle"].power_series_all_data()
    fridge_power = power_elec["fridge"].power_series_all_data()
    dish_washer_power = power_elec["dish washer"].power_series_all_data() 
    microwave_power = power_elec["microwave"].power_series_all_data()

    reference = (len(washer_dryer_power), len(kettle_power), len(fridge_power), len(dish_washer_power), len(microwave_power))
    reference = np.max(reference)

    for app in list(ukdale_appliance_data.keys()):
        
        app_power = np.pad(power_elec[app].power_series_all_data(), (0, reference-len(power_elec[app].power_series_all_data())), 'constant')

        # power = [i for i in power_elec[app].power_series_all_data()]
        # power = power_elec[app].power_series_all_data()
        meter = quantile_filter(ukdale_appliance_data[app]['window'], app_power, p=50)
        state = binarization(meter,ukdale_appliance_data[app]['on_power_threshold'])
        meter = (meter - ukdale_appliance_data[app]['mean'])/ukdale_appliance_data[app]['std']
        targets.append(meter)
        states.append(state)

    mains_denoise = dataset.buildings[1].elec.mains().power_series_all_data()
    print("Before Filter", len(mains_denoise))
    mains_denoise = quantile_filter(10, mains_denoise, 50)

    mains = dataset.buildings[1].elec.mains().power_series_all_data().values-np.percentile(dataset.buildings[1].elec.mains().power_series_all_data().values, 1)
    mains = np.where(mains <mains_denoise, mains_denoise, mains)
    mains = quantile_filter(10, mains, 50)
    mains_denoise = (mains_denoise - 123)/369
    mains = (mains-389)/445
    
    states = np.stack(states).T    
    targets = np.stack(targets).T

    ###
    # mains = int (len(mains)/len(targets))
    # print("Len of Mains Before Downsampling: ", len(mains))
    # print("Mains shape: ", np.shape(mains))
    # mains = block_reduce(mains, block_size=(mains), func=np.mean)
    # print("Len of Mains After Downsampling: ", len(mains))
    ###

    mains = np.resize(mains, len(states))
    mains_denoise = np.resize(mains_denoise, len(states))

    save_path = pathsman.SRC_DIR / f"unetnilm/data/ukdale/{data_type}"
                                    
    del meter, state
    np.save(str(save_path) + "/denoise_inputs.npy", mains_denoise)
    np.save(str(save_path) + "/noise_inputs.npy", mains)
    np.save(str(save_path) + "/targets.npy", targets)
    np.save(str(save_path) + "/states.npy", states)  

if __name__ == "__main__":
    params = [
        ("test", ("2015-06-01", "2015-06-06")),
        ("validation", ("2014-06-01", "2014-06-06")),
        ("training", ("2014-06-01", "2016-06-01"))
    ]

    for data_type in params:
        print(f"PREPROCESS DATA FOR {data_type[0]}, TIME WINDOW OF {data_type[1]}")
        pre_proc_ukdale(data_type, window=data_type[1])
    