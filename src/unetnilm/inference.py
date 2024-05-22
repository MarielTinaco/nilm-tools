import torch
import numpy as np
import pytorch_lightning as pl

from net.model_pl import NILMnet

ckpt_path = "C:/Users/MTinaco/Dev/Solutions/cos-algo-nilm/src/unetnilm/output/2024-02-22_204557071821/checkpoints/ukdale_UNETNiLM_quantiles_multi-appliance/unetnilm-epoch=00-val_F1=0.99.ckpt"

if __name__ == "__main__": 
        
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

        model_name="UNETNiLM" 
        denoise=True
        batch_size = 128 
        epochs = 50
        sequence_length =99 
        sample = None 
        dropout = 0.25 
        data = "ukdale" 
        benchmark="multi-appliance"
        appliance_id = 0
        appliances = list(appliance.keys())
        out_size = 5 
        quantiles=[0.0025,0.1, 0.5, 0.9, 0.975]

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
                        "checkpoint_path" :f"checkpoints/{file_name}/"
                        }

        hparams = NILMnet.add_model_specific_args()
        hparams = vars(hparams.parse_args())
        hparams.update(params)

        model = NILMnet.load_from_checkpoint(ckpt_path, hparams=hparams)

        y = model.predict(model, model.val_dataloader())

        print(y)