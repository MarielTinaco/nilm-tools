# Downloading UKDALE Dataset in H5 format

Create a folder named data from root and copy the UKDALE.h5 file into C:/(path)/(to)/(repo)/data/ukdale/UKDALE.h5

[UKDALE Disaggregated 2015](https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2015/UK-DALE-disaggregated/ukdale.h5.tgz)

[UKDALE Disaggregated 2017](https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.h5.zip)

[UKDALE Main data source](https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic)


# Multi-environment setup for Data preprocessing and Model training

To reconcile conflicts in python toolings for Load disaggregation, a multi-environment setup is employed as a temporary solution. This means that you must switch between two Python environments (Anaconda and Pip) to conduct different stages of load disaggregations

## 1. Data preprocessing on NILMTK using Anaconda environment

NILMTK is a large collection of tools and utilities for Load disaggregation. It is only stable and well-supported on Anaconda

### Setup
1. Setup and install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) environment.
2. Create Anaconda virtual environment using yml file (The name of the environment is found on line 1 of .yml file)
    ```
    conda env create -f environment.yml
    ```
3. Activate virtual environment
    ```
    conda activate <name of environment>
    ```
4. Make sure you are in the root directory
    ```
    (nilmtk-env) C:\(path)\(to)\(directory)\cos-algo-nilm> 
    ```
5. Run Data extractor script
    ```
    python -m scripts.generate_data
    ```

## 2. Model training on UNETNILM

UNETNILM is an open-source project which uses the UNet CNN architecture which is typically used for image/audio segmentation tasks, in Load disaggregation. The early version for pytorch-lightning used in the example repository for UNETNiLM is found on pip. It has only been tested on python 3.8. 

### Setup
1. Install [Python 3.8.10](https://www.python.org/downloads/release/python-3810/) and Setup your virtual environment:
    > If the default global interpreter is already python 3.8:
    ```
    python -m venv venv --prompt nilmenv
    ```
    > If not, do it manually
    ```
    C:/path/to/python/installation/directory/python.exe -m venv venv --prompt nilmenv
    ```
    > You may already use any of your preferred package handler in Python

2. Activate environment
    ```
    C:\(path)\(to)\(directory)\cos-algo-nilm>venv\Scripts\activate
    ```
3. Install dependencies
    - Without CUDA:
    ```
    pip install -r requirements.txt
    ```
    - With CUDA:
    ```
    pip install -r requirements-cu118.txt
    ```

4. Change into src/unetnilm directory
    ```
    cd src\unetnilm
    ```

5. Run experiment.py
    ```
    python experiment.py
    ```