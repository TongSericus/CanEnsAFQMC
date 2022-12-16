# Documentation  

## Introduction 

This webpage contains the details of a finite-temperature canonical ensemble auxiliary-field quantum Monte Carlo (AFQMC) code actively developed in Julia.

In this current version, it can be used to simulate fermions in the Hubbard model in the one and two dimensions and allows for measuring regular observables like energy, momentum distribution and correlation functions as well as quantum-information-related quantities like purity and fidelity. The measurement of Rényi and accessible entanglement entropies are also availble given a bipartition of the lattice, but is not fully optimized.

If you have questions on code usage or bug reports, please contact me at tong_shen1@brown.edu

## Usage

Some example scripts are provided in the [examples](https://github.com/TongSericus/CanEnsAFQMC/tree/main/examples) folder. To enter the module environments and run the scripts, you can execute `julia --project=@. scriptname.jl`

