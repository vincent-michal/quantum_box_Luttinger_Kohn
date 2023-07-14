The script quantum_box_LK.py allows to construct and diagonalize the Luttinger Kohn Hamiltonian for a confined hole (S=3/2). The definition of the model in the continuum can be found in C. Kloeffel et al., Physical Review B 97, 235422 (2018). 

The corresponding discretized model is built using kwant tight-binding open-source software (https://kwant-project.org/). 

Coupling with the magnetic field orbital contribution (vector potential term) is included at first order in the vector potential, with the gauge specified in the script. 

In the file material_parameters.py the parameters are taken from R. Winkler, Spin-orbit Coupling Effects in Two-Dimensional Electron and Hole Systems, Springer (2003).

Vincent Michal, 13th July 2023
