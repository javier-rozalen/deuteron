# Solving the Deuteron

Main program: deuteron_ansatz_wavefunction.py

This program trains an Artificial Neural Network (ANN) to compute a physical wavefunction: the two bound states
of the Deuteron. Starting from a 'blank' ANN with 4Nhid parameters and a single hidden layer, we use it as the trial 
wavefunction in a Rayleigh-Ritz minimisation scheme, with the parameters being the 4Nhid ANN parameters. This 
program is the first step towards the energy minimisation: we train the ANN to take the form of an ansatz physical 
wavefunction to start with, and we do so by maximising the overlap of the ANN and the ansatz function. 

Note: to watch the the function as it is being trained in real-time, use the Spyder editor. 

