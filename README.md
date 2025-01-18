# BGA-Classification

# Overview

This repository contains code to classify individuals based on an AIM set into populations. The code was built with TabPFN (https://github.com/PriorLabs/tabpfn#-license), which has been published in Nature "Accurate predictions on small data with a tabular foundation model" by Hollmann et al.  <br>

This repsository contains <br>

- a python script to use TabPFN for classification of individuals into populations that can be downloaded. You just have to change the input paths to your test and training data (Application_TabPFN.py). <br>
- a python script that was used to run our experiments (run_experiments_cross_validation.py).
- an  Grafical User Interface to use TabPFN for classification of individuals into populations (Streamlit_TabPFN.py). <br> 
- Example Test and Trainingsdata  (data_test.csv, data_train.csv). This data is simulated. <br>
- Example output data. This contains the ROC AUC, the log loss and the accuracy (example_results.json) . Additionally, it contrains the classification of every unclassified individual into a population and the corresponding prediction probabilities (examples_results_individuals.csv). <br>
- the code that was used to prepare the data of our paper "Enhancing BGA Prediction with machine learning" (Extract_EUROPEAN.py, Extract_continents.py) <br>
- the code to plot the confusion matrices (plot_conufsion_matrix.py). <br>


# Usage

To use the graphical user interface, download the repository and install the requirements. Afterwards, type

```bash
streamlit run Streamlit_TabPFN.py

Then, you can use the example test and trainings data or you can upload your own data. <br>

To use the jupyter notebook, just download it. Then, with Jupyter Notebooks, you can change the input paths to your training and test data. <br>
