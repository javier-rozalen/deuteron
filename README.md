# Solving the Deuteron

Main programs: deuteron.py, deuteron_deep.py

This programs train an Artificial Neural Network (ANN) to compute a physical wave function: the two bound states
of the deuteron. Starting from a 'blank' ANN, we use it as the trial wave function in a Rayleigh-Ritz minimisation 
scheme, with the parameters being the ANN parameters and the cost function being the hamiltonian.

Note: to watch the the function as it is being trained in real-time, use the Spyder editor. PyTorch must be installed
in order for the scripts to execute correctly.

