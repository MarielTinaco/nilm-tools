NILM

Install at least Python 3.9 setup

Custom building NILMTK for Windows
- Update nilmtk and nilm_metadata submodules
```
git submodule update --init --recursive
```
- Inside nilmtk, inside setup.py, for contents of variable "install_requires", change packages requirements with strict versioning to minimum version. .i.e. change == to >=. These strict package versioning are required for conda environments.
```
    install_requires=[
        "pandas>=0.25.3",
        "numpy >= 1.13.3",
        "networkx>=2.1",
        "scipy",
        "tables",
        "scikit-learn>=0.21.2",
        "hmmlearn>=0.2.1",
        "pyyaml",
        "matplotlib>=3.1.3",
        "jupyterlab"
    ],
```
- Change into nilm_metadata and run setup to install nilm_metadata first
```
cd nilm_metadata
python setup.py develop
```
- Then change into nilmtk and run setup to install nilmtk (Assuming current working directory is nilm_metadata)
```
cd ..
cd nilmtk
python setup.py develop
```