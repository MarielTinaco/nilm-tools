
import sys, os
import json
import argparse
from pathlib import Path

sys.path.append("../")

from adinilm.io.nilmtk_extractor import NilmtkSubsectionExtractor
from adinilm.utils import paths_manager as pathsman


if __name__ == "__main__":

        parser = argparse.ArgumentParser(prog="NILMTK Subsection Extraction script",
                                         description="""Extracts subsection of NILMTK dataset
                                                        into generic Pandas dataframe which can
                                                        optionally be saved into a Python version
                                                        agnostic pickle file
                                                     """,
                                         epilog="\n")
        parser.add_argument('--config', '-c')
        args = parser.parse_args()


        CONFIG = Path(args.config)
        with open(CONFIG, "r") as conf:
                config = json.load(conf)
        
        for key, value in config.items():
                ext = NilmtkSubsectionExtractor(dataset_path=pathsman.UKDALE_H5_PATH,
                                                params=value, match_timeframes=True)

                data_dir = pathsman.DATA_DIR
                data_dir.mkdir(exist_ok=True)

                nilmtk_data_dir = data_dir / "NILMTK"
                nilmtk_data_dir.mkdir(exist_ok=True)

                raw_nilmtk_data_dir = nilmtk_data_dir / "raw"
                raw_nilmtk_data_dir.mkdir(exist_ok=True)

                split = raw_nilmtk_data_dir / f"{key}.pkl"

                ext.df.to_pickle(split)