
import sys, os
import argparse
from pathlib import Path

sys.path.append("../")

from adinilm.io.nilmtk_extractor import NilmtkSubsectionExtractor
from adinilm.utils import paths_manager as pathsman

FRIDGE_FREEZER = "fridge freezer"
KETTLE = "kettle"
DISHWASHER = "dish washer"
MICROWAVE = "microwave"
WASHER_DRYER = "washer dryer"

applist = [FRIDGE_FREEZER, KETTLE, DISHWASHER, MICROWAVE, WASHER_DRYER]

if __name__ == "__main__":
        parser = argparse.ArgumentParser(prog="NILMTK Subsection Extraction script",
                                         description="""Extracts subsection of NILMTK dataset
                                                        into generic Pandas dataframe which can
                                                        optionally be saved into a Python version
                                                        agnostic pickle file
                                                     """,
                                         epilog="\n")

        parser.add_argument('-n', '--filename')
        args = parser.parse_args()

        extr = NilmtkSubsectionExtractor(dataset_path=pathsman.UKDALE_H5_PATH,
                                         params={"start_time" : "2015-01-01",
                                                 "end_time" : "2015-01-02",
                                                 "building" : 1,
                                                 "appliances" : applist,
                                                 "sample_period" : 1,
                                                 "resample" : True})

        filename = Path(args.filename)

        if not filename.parent.exists():
               os.mkdir(filename.parent) 

        extr.to_pickle(args.filename)