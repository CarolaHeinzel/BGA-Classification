# BGA-Classification

# Overview

This repository contains code to classify individuals based on an AIM set into populations. The code was built with [TabPFN](https://github.com/PriorLabs/tabpfn#-license), which has been published in [Nature]() "[Accurate predictions on small data with a tabular foundation model] by Hollmann et al.  <br>

This repsository contains:

* `Application_TabPFN.py`: to use TabPFN for classification of individuals into populations that can be downloaded. You just have to change the input paths to your test and training data (Application_TabPFN.py).
* `CrossValidation_Code/run_experiments.py`: a python script that was used to run our experiments.
* `Streamlit_TabPFN.py`: a  Grafical User Interface to use TabPFN for classification of individuals into populations, which runs in the Browser.  
* `data_test.csv`, `data_train.csv`: Example test and training data. This is a subset of the data published as a Supplemental (`1-s2.0-S1872497323000285-mmc5.xlsx`) of [Ruiz-Ramirez et al](https://pubmed.ncbi.nlm.nih.gov/36917866/).
- `prediction_prob.csv`: Example output prediction probabilites. <br>
* `Extract_EUROPEAN.py`, `Extract_continents.py`: the code that was used to prepare the data of our paper [Enhancing BGA Prediction with machine learning](xxx).
* `Cross_Validation_Code/Confusion_matrix.py`: the code to plot the confusion matrices. 
* `Cross_Validation_Code/run_plotting.py`: python code to plot the mean of ROC AUC, accuracy and logloss (or any other metric) and its confidence interval.
* `Cross_Validation_Code/SNIPPER_CrossValidation.R`: Code for the Cross Validation with SNIPPER
* `Cross_Validation_Code/PLS_DA_CrossValidation.py`: Code for the Cross Validation with PLS-DA
* `Cross_Validation_Code/Admixture_Model_CrossValidation.py`:  Code for the Cross Validation with the Admixture Model
* `Cross_Validation_Code/TabPFN_RandomForest_XGBoost_CrossValidation.py`:  Code for the Cross Validation with TabPFN, XGBoost, RandomForest


# Usage: Web Tool

See [here](https://bga-classification-xjmafspgoed8xh65fyfddh.streamlit.app/) for an online version of the Grapical User Interface. Note that there are no data protection rules on the server where this interface runs.

# Usage: Local Graphical User Interface

To use the graphical user interface, download the repository and install the requirements using
```
pip install -r requirements.txt
```
Afterwards, type
```bash
streamlit run Streamlit_TabPFN.py
```

Then, your browser should open and you can use the example test and trainings data or you can upload your own data. <br>

# Usage: Command Line

With
```
python Application_TabPFN.py
```
a TabPFN run is performed on `train_data_eur.csv` and `test_data_eur.csv`. Change the file according to your needs if it should run on a different dataset.
