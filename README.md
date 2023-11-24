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
