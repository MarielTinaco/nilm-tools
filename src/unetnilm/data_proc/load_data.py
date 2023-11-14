import sys
import pandas as pd
import numpy as np
import nilmtk

sys.path.append('../')
from src.utils import paths_manager as pathsman

data_type = ("training")
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

ukdale_appliance_data = {
    "kettle": {
        "mean": 700,
        "std": 1000,
        'window':10,
        'on_power_threshold': 2000,
        'max_on_power': 3998
    },
    "fridge": {
        "mean": 200,
        "std": 400,
        "window":50,
        'on_power_threshold': 50,
    },
    "dish washer": {
        "mean": 700,
        "std": 700,
        "window":50,
        'on_power_threshold': 10
    },
    "washer dryer": {
        "mean": 400,
        "std": 700,
        "window":50,
        'on_power_threshold': 20,
        'max_on_power': 3999
    },
    "microwave": {
        "mean": 500,
        "std": 800,
        "window":10,
        'on_power_threshold': 200,
    },
}

def pre_proc_ukdale(src_dir, window):
    targets = []
    states = [] 
    dataset = nilmtk.DataSet(pathsman.UKDALE_H5_PATH)
    dataset.set_window(*window)
    power_elec = dataset.buildings[1].elec
    
    for app in list(ukdale_appliance_data.keys()):
        power = [i for i in power_elec[app].power_series_all_data()]
        meter = quantile_filter(ukdale_appliance_data[app]['window'], power, p=50)
        state = binarization(meter,ukdale_appliance_data[app]['on_power_threshold'])
        meter = (meter - ukdale_appliance_data[app]['mean'])/ukdale_appliance_data[app]['std']
        targets.append(meter)
        states.append(state)
        
    mains_denoise = dataset.buildings[1].elec.mains().power_series_all_data()
    print (mains_denoise)
    mains_denoise = quantile_filter(10, mains_denoise, 50)

    mains = dataset.buildings[1].elec.mains().power_series_all_data().values-np.percentile(dataset.buildings[1].elec.mains().power_series_all_data().values, 1)
    mains = np.where(mains <mains_denoise, mains_denoise, mains)
    mains = quantile_filter(10, mains, 50)
    mains_denoise = (mains_denoise - 123)/369
    mains = (mains - 389)/445

    states = np.hstack(states).T
    targets = np.hstack(targets).T

    del power, meter, state
    np.save(save_path+f"/ukdale/{data_type}/denoise_inputs.npy", mains_denoise)
    np.save(save_path+f"/ukdale/{data_type}/noise_inputs.npy", mains)
    np.save(save_path+f"/ukdale/{data_type}/targets.npy", targets)
    np.save(save_path+f"/ukdale/{data_type}/states.npy", states)  

if __name__ == "__main__":
    for data_type in ["test", "validation", "training"]:
        print(f"PREPROCESS DATA FOR {data_type}")
        pre_proc_ukdale(data_type)
    