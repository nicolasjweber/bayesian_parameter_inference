# Data-Driven Bayesian Parameter Estimation with Neural Networks for Power Grid Frequency
This repository contains the code for the Bachelor's Thesis "Data-Driven Bayesian Parameter Estimation with Neural Networks for Power Grid Frequency" at KIT.

We recommend using `conda` for creating a virtual environment as follows:

    conda create -n sbi_env python=3.9 
    conda activate sbi_env
    pip install -r requirements.txt
  
  
The enumeration of the files reflects their recommended execution order.
Most importantly, ensure that the respective posterior estimates have been calculated via `01_train.py` before executing the code of the evaluation part of the thesis.  

### Datasets
The datasets used in this thesis can be obtained from:
  
 - Continental European Grid: [Pre-Processed Power Grid Frequency Time Series](https://zenodo.org/records/5105820)
 - Balearic Grid: [Open Access Power-Grid Frequency Database](https://osf.io/m43tg/) and [Red Eléctrica Dashboard](https://demanda.ree.es/visiona/baleares/baleares5m/tablas/2019-11-6/1)
 
The daytime-specific parameter ranges used in the thesis are extracted from the [supplementary data](https://github.com/johkruse/PIML-for-grid-frequency-modelling) to the paper "Physics-informed machine learning for power grid frequency modelling" by Kruse et al. The notebook `00_methodology_daytime_specific.ipynb` converts the extracted data to a CSV file later used by `01_train.py`.
