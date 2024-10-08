from pathlib import Path

from nilm.unetnilm.data_proc.load_data import pre_proc_ukdale, pre_proc_ukdale_nilmtk

# params = [
#         ("test", ("2014-06-01", "2014-06-06")),
#         ("training", ("2014-06-01", "2014-06-15")),
#         ("validation", ("2014-07-01", "2014-07-06"))
#         ]

# for data_type in params:
#         print(f"PREPROCESS DATA FOR {data_type[0]}, TIME WINDOW OF {data_type[1]}")
#         pre_proc_ukdale(data_type[0], window=data_type[1])

params = [
        ("training", {"start": "2015-01-01", "end": "2016-06-30"}),
]

for param in params:
        print(f"PREPROCESS DATA FOR {param[0]}, TIME WINDOW OF {param[1]}")
        directory = param[0]
        
        main_path = Path("src") / "unetnilm" / "data" / "ukdale" / directory
        main_path.mkdir(parents=True, exist_ok=True)

        open(main_path / "targets.npy", "a").close()
        open(main_path / "states.npy", "a").close()
        open(main_path / "noise_inputs.npy", "a").close()
        open(main_path / "denoise_inputs.npy", "a").close()
        
        pre_proc_ukdale_nilmtk(param[0], timeframe=param[1])

