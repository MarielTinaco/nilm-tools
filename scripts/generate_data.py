from src.unetnilm.data_proc.load_data import pre_proc_ukdale, pre_proc_ukdale_nilmtk

# params = [
#         ("test", ("2014-06-01", "2014-06-06")),
#         ("training", ("2014-06-01", "2014-06-15")),
#         ("validation", ("2014-07-01", "2014-07-06"))
#         ]

# for data_type in params:
#         print(f"PREPROCESS DATA FOR {data_type[0]}, TIME WINDOW OF {data_type[1]}")
#         pre_proc_ukdale(data_type[0], window=data_type[1])

params = [
        ("training", {"start": "2014-05-25", "end": "2014-05-30"}),
        ("test", {"start": "2014-06-01", "end": "2014-06-05"}),
        ("validation", {"start": "2014-08-30", "end": "2014-09-05"})
]

for param in params:
        print(f"PREPROCESS DATA FOR {param[0]}, TIME WINDOW OF {param[1]}")
        pre_proc_ukdale_nilmtk(param[0], timeframe=param[1])

