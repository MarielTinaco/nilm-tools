from pathlib import Path

import numpy as np
import pandas as pd
import nilmtk

class NilmtkSubsectionExtractor:

        COLUMNS = ["power_series", "on_power_threshold", "activations"]

        def __init__(self, dataset_path, params=None):
                self.dataset_path = Path(dataset_path)
                self.params = params
                self.df = pd.DataFrame(columns=self.COLUMNS)
                if self.params:
                        self.subsect(self.params)

                setattr(self, "to_pickle", self.df.to_pickle)
                setattr(self, "loc", self.df.loc)
                setattr(self, "iloc", self.df.iloc)

        def subsect(self, params):
                ds = nilmtk.DataSet(self.dataset_path)
                ds.set_window(start=params["subsection"]["start_time"], end=params["subsection"]["end_time"])
                elec = ds.buildings[params["subsection"]["building"]].elec

                apps_data = []
                for app in params["appliances"]:
                        per_app = []
                        per_app.append(elec[app].power_series_all_data(sample_period=params["preprocessing"]["sampling"]["sample_period"],
                                                                       resample=params["preprocessing"]["sampling"]["resample"]))
                        per_app.append(elec[app].on_power_threshold())
                        per_app.append(elec[app].get_activations())
                        apps_data.append(per_app)

                apps_data.append([elec.mains().power_series_all_data(sample_period=params["preprocessing"]["sampling"]["sample_period"],
                                                                     resample=params["preprocessing"]["sampling"]["resample"]),
                                  np.NaN,
                                  np.NaN])
                self.df = pd.DataFrame(apps_data, index=params["appliances"] + ["site meter"], columns=self.COLUMNS)
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

        # print(extr.df.loc["fridge freezer", "power_series"])
        # print(extr.df.loc["site meter", "on_power_threshold"])