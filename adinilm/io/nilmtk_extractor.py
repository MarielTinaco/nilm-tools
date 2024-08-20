from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import nilmtk

class NilmtkSubsectionExtractor:

        COLUMNS = ["power_series", "on_power_threshold", "activations"]

        def __init__(self, dataset_path, params=None, match_timeframes=False):
                self.dataset_path = Path(dataset_path)
                self.params = params
                self.df = pd.DataFrame(columns=self.COLUMNS)
                if self.params:
                        self.subsect(self.params, match_timeframes=match_timeframes)

                setattr(self, "to_pickle", self.df.to_pickle)
                setattr(self, "loc", self.df.loc)
                setattr(self, "iloc", self.df.iloc)

        def subsect(self, params, match_timeframes=False):
                ds = nilmtk.DataSet(self.dataset_path)
                ds.set_window(start=params["subsection"]["start_time"], end=params["subsection"]["end_time"])
                elec = ds.buildings[params["subsection"]["building"]].elec

                if match_timeframes:
                        submeter_sample_period = 1
                        sitemeter_sample_period = params["preprocessing"]["sampling"]["sample_period"]
                        resample = True
                else:
                        submeter_sample_period = params["preprocessing"]["sampling"]["sample_period"]
                        sitemeter_sample_period = params["preprocessing"]["sampling"]["sample_period"] 
                        resample = params["preprocessing"]["sampling"]["resample"]

                apps_data = []
                for app in params["appliances"]:
                        per_app = []
                        per_app.append(elec[app].power_series_all_data(sample_period=submeter_sample_period,
                                                                resample=resample))
                        per_app.append(elec[app].on_power_threshold())
                        per_app.append(elec[app].get_activations())
                        apps_data.append(per_app)

                apps_data.append([elec.mains().power_series_all_data(sample_period=sitemeter_sample_period,
                                                                resample=resample), np.NaN, np.NaN])

                self.df = pd.DataFrame(apps_data, index=params["appliances"] + ["site meter"], columns=self.COLUMNS)

                if match_timeframes:
                            self.match_timeframes_to_site_meter(energy_type="power_series")

                return self

        def match_timeframes_to_site_meter(self, energy_type = "power_series"):

                site_meter_power_series = self.df.loc["site meter", energy_type]

                submeter_power_series = self.df.loc[self.params["appliances"], energy_type]

                for key, submeter in submeter_power_series.items():
                        submeter.set_axis(submeter.index.floor(freq='s'))
                        filtered_site_meter_pow_series = site_meter_power_series[site_meter_power_series.index > submeter.index[0]]
                        site_meter_index = filtered_site_meter_pow_series.index.floor(freq='s')
                        submeter = submeter[site_meter_index]
                        self.df.at[key, 'power_series'] = submeter

                # for key, submeter in submeter_power_series.items():
                #         submeter_iter = iter(submeter.items())
                #         new_submeter_idx = []
                #         new_submeter_vals = []

                #         for timestamp in site_meter_index:
                #                 ts, val = next(submeter_iter)
                #                 if timestamp < ts:
                #                         continue
                #                 else:
                #                         while (ts.floor(freq='s') != timestamp):
                #                                 ts, val = next(submeter_iter)
                #                         else:
                #                                 new_submeter_idx.append(timestamp)
                #                                 new_submeter_vals.append(val)

                #         matched_submeter = pd.Series(data=new_submeter_vals, index=new_submeter_idx)
                #         self.df.at[key, 'power_series'] = matched_submeter


if __name__ == "__main__":

        from adinilm.utils import paths_manager as pathsman

        FRIDGE_FREEZER = "fridge freezer"
        KETTLE = "kettle"
        DISHWASHER = "dish washer"
        MICROWAVE = "microwave"
        WASHER_DRYER = "washer dryer"

        applist = [FRIDGE_FREEZER, KETTLE, DISHWASHER, MICROWAVE, WASHER_DRYER]

        UKDALE_BUILDING = 1
        UKDALE_START_TIME = "2015-01-01"
        # END_TIME = "2015-01-15"
        UKDALE_END_TIME = "2015-01-02"

        info = {
                "subsection" : {
                        "building" : UKDALE_BUILDING,
                        "start_time" : UKDALE_START_TIME,
                        "end_time": UKDALE_END_TIME
                },
                "preprocessing" : {
                        "sampling" : {
                                "sample_period" : 6,
                                "resample" : True
                        },
                        "normalization" : {
                                "mode" : "minmax",
                                "scope" : "local"
                        }
                },
                "appliances" : [FRIDGE_FREEZER, WASHER_DRYER, KETTLE, DISHWASHER, MICROWAVE],
                "appliance_data" : {
                        FRIDGE_FREEZER : {
                                "window" : 50,
                                "min" : 0,
                                "max" : 0,
                                "on_power_threshold" : 10,
                        },
                        WASHER_DRYER : {               
                                "window" : 50,
                                "min" : 0,
                                "max" : 0,
                                "on_power_threshold" : 10,
                        },
                        KETTLE : {
                                "window" : 50,
                                "min" : 0,
                                "max" : 0,
                                "on_power_threshold" : 10,
                        },
                        DISHWASHER : {
                                "window" : 10,
                                "min" : 0,
                                "max" : 0,
                                "on_power_threshold" : 10,
                        },
                        MICROWAVE : {
                                "window" : 50,
                                "min" : 0,
                                "max" : 0,
                                "on_power_threshold" : 10,
                        }
                }
        }

        extr = NilmtkSubsectionExtractor(dataset_path=pathsman.UKDALE_H5_PATH,
                                         params=info,
                                         match_timeframes=True)
