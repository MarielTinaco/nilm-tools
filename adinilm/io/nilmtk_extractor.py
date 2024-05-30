from pathlib import Path

import pandas as pd
import nilmtk

class NilmtkSubsectionExtractor:

        def __init__(self, dataset_path, params=None):
                self.dataset_path = Path(dataset_path)
                self.params = params
                self.df = pd.DataFrame(columns=["power_series", "on_power_threshold"])
                if self.params:
                        self.subsect(self.params)

        def subsect(self, params):
                ds = nilmtk.DataSet(self.dataset_path)
                ds.set_window(start=params["start_time"], end=params["end_time"])
                elec = ds.buildings[params["building"]].elec
                apps_data = [[elec[app].power_series_all_data(sample_period=params["sample_period"],
                                                              resample=params["resample"]),
                              elec[app].on_power_threshold()]
                              for app in params["appliances"]]
                self.df = pd.DataFrame(apps_data, index=params["appliances"], columns=["power_series", "on_power_threshold"])
                return self

        def to_pickle(self, filename):
                self.df.to_pickle(filename)
                return self

if __name__ == "__main__":

        from adinilm.utils import paths_manager as pathsman

        FRIDGE_FREEZER = "fridge freezer"
        KETTLE = "kettle"
        DISHWASHER = "dish washer"
        MICROWAVE = "microwave"
        WASHER_DRYER = "washer dryer"

        applist = [FRIDGE_FREEZER, KETTLE, DISHWASHER, MICROWAVE, WASHER_DRYER]

        extr = NilmtkSubsectionExtractor(dataset_path=pathsman.UKDALE_H5_PATH,
                                         params={"start_time" : "2015-01-01",
                                                 "end_time" : "2015-01-02",
                                                 "building" : 1,
                                                 "appliances" : applist,
                                                 "sample_period" : 1,
                                                 "resample" : True})

        print(extr.df.loc["fridge freezer", "power_series"])