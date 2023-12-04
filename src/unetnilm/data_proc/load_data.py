import sys
import pandas as pd
import numpy as np
from typing import Union, Tuple, Dict
from pathlib import Path
import nilmtk   
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

sys.path.append('../../')
from src.utils import paths_manager as pathsman
from src.unetnilm.utils.mappings import ukdale_appliance_data


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
        state = binarization(meter,power_elec[app].on_power_threshold())
        meter = (meter - ukdale_appliance_data[app]['mean'])/ukdale_appliance_data[app]['std']
        targets.append(meter)
        states.append(state)

    mains_denoise = dataset.buildings[1].elec.mains().power_series_all_data()
    print("Before Filter", len(mains_denoise))
    mains_denoise = quantile_filter(10, mains_denoise, 50)

    mains = dataset.buildings[1].elec.mains().power_series_all_data().values-np.percentile(dataset.buildings[1].elec.mains().power_series_all_data().values, 1)
    mains = np.where(mains <mains_denoise, mains_denoise, mains)
    mains = quantile_filter(10, mains, 50)
    mains_denoise = (mains_denoise - mains_denoise.mean())/mains_denoise.std()
    mains = (mains-mains.mean())/mains.std()
    
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

def pre_proc_ukdale_nilmtk(data_type, timeframe : Union[Tuple, Dict], building : int = 1):
    targets = []
    states = [] 
    dataset = nilmtk.DataSet(pathsman.UKDALE_H5_PATH)
    
    appliance = {
        "fridge" : {
            "window" : 50,
            "mean" : dataset.buildings[building].elec["fridge"].power_series_all_data().mean(),
            "std" : dataset.buildings[building].elec["fridge"].power_series_all_data().std(),
        },
        "boiler" : {
            "window" : 50,
            "mean" : dataset.buildings[building].elec["boiler"].power_series_all_data().mean(),
            "std" : dataset.buildings[building].elec["boiler"].power_series_all_data().std()
        },
        "washer dryer" : {
            "window" : 50,
            "mean" : dataset.buildings[building].elec["washer dryer"].power_series_all_data().mean(),
            "std" : dataset.buildings[building].elec["washer dryer"].power_series_all_data().std()
        },
        "HTPC" : {
            "window" : 50,
            "mean" : dataset.buildings[building].elec["HTPC"].power_series_all_data().mean(),
            "std" : dataset.buildings[building].elec["HTPC"].power_series_all_data().std()
        },
        "dish washer" : {
            "window" : 50,
            "mean" : dataset.buildings[building].elec["dish washer"].power_series_all_data().mean(),
            "std" : dataset.buildings[building].elec["dish washer"].power_series_all_data().std()
        },
        "microwave" : {
            "window" : 10,
            "mean" : dataset.buildings[building].elec["microwave"].power_series_all_data().mean(),
            "std" : dataset.buildings[building].elec["microwave"].power_series_all_data().std()
        }
    }

    mains_mean = dataset.buildings[building].elec.mains().power_series_all_data().mean()
    mains_std = dataset.buildings[building].elec.mains().power_series_all_data().std()

    if isinstance(timeframe, Tuple):
        dataset.set_window(*timeframe)
    elif isinstance(timeframe, dict):
        dataset.set_window(**timeframe)

    power_elec = dataset.buildings[building].elec

    indices = [power_elec[app].power_series_all_data().index for app in appliance.keys()]
    sorted_indices = sorted(indices, key=len)

    main_index = sorted_indices[0]
    reduced_power_series_list = []

    for app in appliance.keys():
            power_series = power_elec[app].power_series_all_data()
            reduced_power_series = power_series[power_series.index.get_indexer(main_index, method="nearest")]
            reduced_power_series_list.append(reduced_power_series)
            # reduced_power_series_list.append(reduced_power_series)
            meter = quantile_filter(appliance[app]["window"], reduced_power_series, p=50)
            state = binarization(meter, power_elec[app].on_power_threshold())
            meter = (meter - appliance[app]['mean'])/appliance[app]['std']
            targets.append(meter)
            states.append(state)

    # mains_series = power_elec.mains().power_series_all_data()
    # reduced_main_power_series = mains_series[mains_series.index.get_indexer(main_index, method="nearest")]
    
    mains_series = reduced_power_series_list[0]
    for i in reduced_power_series_list[1:]:
        mains_series += i.values
    reduced_main_power_series = mains_series

    tv_power_series = power_elec["television"].power_series_all_data()
    reduced_tv_power_series = tv_power_series[tv_power_series.index.get_indexer(main_index, method="nearest")]

    reduced_main_power_series = reduced_main_power_series + reduced_tv_power_series.values

    mains_denoise = quantile_filter(10, reduced_main_power_series, 50)

    mains = reduced_main_power_series.values-np.percentile(reduced_main_power_series.values, 1)
    mains = np.where(mains < mains_denoise, mains_denoise, mains)
    mains = quantile_filter(10, mains, 50)
    
    norm_mains_denoise = (mains_denoise - mains_denoise.mean()) / mains_denoise.std()
    norm_mains = (mains - mains_mean) / mains_std

    states = np.stack(states).T
    targets = np.stack(targets).T

    save_path = pathsman.SRC_DIR / f"unetnilm/data/ukdale/{data_type}"
                                    
    del meter, state
    np.save(str(save_path) + "/denoise_inputs.npy", norm_mains_denoise)
    np.save(str(save_path) + "/noise_inputs.npy", norm_mains)
    np.save(str(save_path) + "/targets.npy", targets)
    np.save(str(save_path) + "/states.npy", states)