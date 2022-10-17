# Solving the deuteron

## Requirements
The machine learning part of the code in the files above is written in PyTorch. It does not come with the default Python 3 installation; to install it, go to [Official PyTorch page](https://pytorch.org/get-started/locally/) or type:

`pip3 install torch`

Also, the progress bar `tqdm` is used. To install it:

`pip3 install tqdm` 


## Model training guide
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The steps below are a guide through the code files in this repository intended to go over the whole training process of the Neural Networks.

### Step 0. Architecture.
We choose an architecture among the four appearing in the article (1sc, 2sc, 1sd, 2sd). 

### Step 1. Pretraining.
We open 'pretraining/pretraining.py', which trains an ANN to match two wave function-like functions. We set the parameters at the beginning of the file to our preference, especially the architecture chosen in Step 0. This will store the models and the plots in 'saved_models/pretraining/nlayers?/?/Sigmoid/lr?/'. 

### Step 2. Training.
We go back to the main folder and open 'deuteron.py', which trains the ANN to minimise the energy. A loop is already programmed that will sequentially load the pretrained models (Step 1). The code allows to split the total list of pretrained models into batches so as to seize parallelized computing (clusters). By the end of this step we already have a fully trained model!

Note: the steps below are dedicated to error analysis and plotting.

### Step 3. Error analysis.
We open 'error_analysis/error_measure.py', file with which we can automatically compute the errors of the trained models. A loop is already programmed that will sequentially load the trained models (Step 2). This will save E, Ks, Kd, Pd with their corresponding errors to error_data/nlayers?/nhid?.txt. 

### Step 4. Filtering the good runs.
We open error_analysis/filter.py and adjust the initial parameters. This program filters the trained models and selects the ones that match our criteria (defined via the initial parameters). The selected runs are copied in a folder named filtered_runs. This step is meant to make plotting easier.

### Step 5. Plotting.
For a complete plot including data of the four architectures, we repeat Steps 1 to 4 for all architectures (Step 0). After that, we open 'plotting/energy_plot.py'. We set the initial parameters and we run the file. Text files under plotting/plotting_data/ with further errors will be generated in the process. Once the program is done, the resulting plot will be stored in plotting/saved_plots/energy_plot.pdf

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

