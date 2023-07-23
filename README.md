# Documentation  
[![Build Status](https://github.com/TongSericus/CanEnsAFQMC/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/TongSericus/CanEnsAFQMC/actions/workflows/CI.yml?query=branch%3Amaster)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2212.08654-B31B1B.svg)](https://arxiv.org/abs/2212.08654)

## Introduction 

This webpage contains the details of a finite-temperature canonical ensemble auxiliary-field quantum Monte Carlo (AFQMC) code actively developed in Julia.

In this current version, it can be used to simulate fermions in the Hubbard model in the one and two dimensions and allows for measuring observables like energy, momentum distribution and correlation functions as well as quantum-information-related quantities like [purity](https://github.com/TongSericus/CanEnsAFQMC/tree/master/examples/job_purity_ce.jl). The measurement of Rényi and accessible entanglement entropies are also availble given a bipartition of the lattice, but is not fully optimized.

If you have questions on code usage or bug reports, please contact me at tong_shen1@brown.edu

## Usage

Some example scripts are provided in the [examples](https://github.com/TongSericus/CanEnsAFQMC/tree/master/examples) folder. To enter the module environments and run the scripts (instantiation is required), you can execute `julia --project=@. scriptname.jl`

