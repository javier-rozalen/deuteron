# Solving the Deuteron

GENERAL GUIDELINES

Side note 1: the ? sign found throughout this document is to be substituded with the network architecture (and hyperparameters) chosen in Step 0.
Side note 2: there are multiple adjustable parameters in each code file, all of them are explained at the beginning of their respective file.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Step 0. We choose an architecture among the four appearing in the article (1sc, 2sc, 1sd, 2sd). 
Step 1. We open 'pretraining/pretraining.py', which trains an ANN to match two wave function-like functions. We set the parameters at the beginning of the file to our preference, especially the architecture chosen in Step 0. This will store the models and the plots in 'saved_models/pretraining/nlayers?/?/Sigmoid/lr?/'. 

Step 2. We go back to the main folder and open 'deuteron.py', which trains the ANN to minimise the energy. A loop is already programmed that will sequentially load the pretrained models (Step 1). The code allows to split the total list of pretrained models into batches so as to seize parallelized computing (clusters). 
By the end of this step we already have a fully trained model!

Note: All steps below are devoted to error analysis and plotting.

Step 3. We open 'error_analysis/error_measure.py', file with which we can automatically compute the errors of the trained models. A loop is already programmed that will sequentially load the trained models (Step 2).
This will save E, Ks, Kd, Pd with their corresponding errors to error_data/nlayers?/nhid?.txt. 

Step 4. We open filter.py and adjust the initial parameters. This program filters the trained models and selects the ones that match our criteria (defined via the initial parameters). The selected runs are copied in a folder named filtered_runs. This step is thought to make plotting easier.

Step 5. We open 'error_analysis/plots.py'. We set Nhid(Hd) accordingly, 'save_individual_plot_data'=True, and 'filename' and 'filename2' accordingly, and we run.
This will compute all the types of error and append() them to the file 'error_analysis/nlayers?/joint_graph.txt'. 

Step 6. In the same script, we set 'save_individual_plot_data'=False and we run the script. This will produce the final plot.

