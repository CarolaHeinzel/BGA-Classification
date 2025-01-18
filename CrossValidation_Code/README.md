# Code for Experiments


## Install 
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install uv
uv pip install -r requirements.txt
```

## Usage 

1. Prepare the data using `run_convert_excel_data.py` script. 
    * This has already been done for the repo and you can skip this step.
2. Run the `run_experiment.py` script to run the experiments
   * This has already been done for the repo and you can skip this step.
   * If you want to run new models, you can use the code from run experiment to run the models and combine the outputs.
3. Run the `run_plotting.py` script to plot the results
   * You can find the plots in the `plots` directory.
   * You can also find the results in the `results` directory.
     * We have the score results (`results.csv`) and the model outputs (`raw_predictions.json`) per fold.