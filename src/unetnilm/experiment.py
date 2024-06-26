from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import numpy as np
from net.model_pl import NILMnet
from net.utils import DictLogger
from pathlib import Path
import pytorch_lightning as pl
from net.utils import get_latest_checkpoint
from utils.utils import set_seed, get_device
from IPython.display import clear_output

from datetime import datetime

import sys
from argparse import ArgumentParser
set_seed(seed=7777)
device =  get_device()

current_time = str(datetime.now()).replace(" ","_").replace(".","").replace(":","")

class NILMExperiment(object):

    def __init__(self, params):
        """
        Parameters to be specified for the model
        """
        self.MODEL_NAME = params.get('model_name',"CNNModel")
        self.logs_path =params.get('log_path',f"output/{current_time}/logs/")
        self.checkpoint_path =params.get('checkpoint_path',f"output/{current_time}/checkpoints/")
        self.results_path = params.get('results_path',f"output/{current_time}/results/")
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10 )
        self.batch_size = params.get('batch_size',128)
        self.dropout = params.get('dropout', 0.1)
        self.params = params
        
        #create files
        logs = Path(self.logs_path )
        checkpoints = Path(self.checkpoint_path)
        results = Path(self.results_path)
        logs.mkdir(parents=True, exist_ok=True)
        checkpoints.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        

    def fit(self):
        file_name = self.params['file_name']
        self.arch = file_name
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=self.checkpoint_path,filename='unetnilm-{epoch:02d}-{val_F1:.2f}', monitor='val_F1', mode="max",save_top_k=-1)
        early_stopping = pl.callbacks.EarlyStopping(monitor='val_F1', min_delta=1e-4, patience=20, mode="max")
        logger = DictLogger(self.logs_path, name=file_name, version=self.params['exp_name'])
        trainer = pl.Trainer(
                    logger = logger,
                    gradient_clip_val=self.params['clip_value'],
                    # checkpoint_callback=checkpoint_callback,
                    callbacks=[checkpoint_callback, early_stopping],
                    max_epochs=self.params['n_epochs'],
                    gpus=-1 if torch.cuda.is_available() else None,
                    # callbacks=early_stopping,
                    resume_from_checkpoint=get_latest_checkpoint(self.checkpoint_path)
                     )
        
        self.hparams = NILMnet.add_model_specific_args()
        self.hparams = vars(self.hparams.parse_args())
        self.hparams.update(self.params)
        model = NILMnet(self.hparams)
        print(f"fit model for { file_name}")
        trainer.fit(model)
        # (1) load the best checkpoint automatically (lightning tracks this for you)
        results=trainer.test()
        clear_output()
        print(results[0]['app_results'])
        
        
        results_path = f"{self.results_path}{file_name}"
        return results[0], results_path
        
def run_experiments(model_name="CNN1D", denoise=True,
                     batch_size = 128, epochs = 50,
                    sequence_length =99, sample = None, 
                    dropout = 0.25, data = "ukdale", 
                    benchmark="single-appliance",
                    appliance_id = 0,
                    appliances = ["FRZ"],
                    out_size = 5, quantiles=[0.0025,0.1, 0.5, 0.9, 0.975]):        
    exp_name = f"{data}_{model_name}_quantiles" if len(quantiles)>1 else "{data}_{model_name}"
    if benchmark=="single-appliance":
        file_name = f"{exp_name}_single-appliance_{appliances[0]}"
    else:
        file_name = f"{exp_name}_multi-appliance"      
    
    params = {'n_epochs':epochs,'batch_size':batch_size,
                'sequence_length':sequence_length,
                'model_name':model_name,
                'dropout':dropout,
                'exp_name':exp_name,
                'benchmark':benchmark,
                'clip_value':10,
                'sample':sample,
                'out_size':out_size,
                'appliance_id':appliance_id,
                'appliances':appliances,
                'out_size':len(appliances),
                'data_path':"data",
                'data':data,
                'quantiles':quantiles,
                "denoise":denoise,
                'file_name':file_name,
                "checkpoint_path" :f"output/{current_time}/checkpoints/{file_name}/"
                }
    exp = NILMExperiment(params)
    results, results_path=exp.fit()
   
    return results, results_path

if __name__ == "__main__": 
    sample=None
    epochs=30
            
    appliance = {
        "fridge" : {
            "window" : 50,
        },
        "washer dryer" : {
            "window" : 50,
        },
        "kettle" : {
            "window" : 10,
        },
        "dish washer" : {
            "window" : 50,
        },
        "microwave" : {
            "window" : 10,
        }
    }

    for data in ["ukdale"]:
        # for model_name in ["UNETNiLM", "CNN1D"]:
        for model_name in ["UNETNiLM"]:
            results = {}
            results, save_path=run_experiments(model_name=model_name, data = data, 
                                sample=sample, epochs=epochs, appliances=list(appliance.keys()),
                                appliance_id=None, benchmark="multi-appliance")  
            np.save(save_path+"results.npy", results)                        
    
    # for data in ["ukdale"]:
    #     for model_name in ["CNN1D", "UNETNiLM"]:
    #         results = {}
    #         for idx, app in enumerate(list(appliance.keys())):
    #             result, save_path=run_experiments(model_name=model_name, data = data, 
    #                             sample=sample, epochs=epochs, appliances=[app],
    #                             appliance_id=idx, benchmark="single-appliance")  
    #             results[app]=result
    #         np.save(save_path+"results.npy", results)
