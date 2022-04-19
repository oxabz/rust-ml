# Data Science with rust  

This repo is a place to store my experiment with tch-rs evcxr and other related datascience stuff in rust.

## Structure : 

It is structured as a cargo workspace split between different crates

```sh
.
├── README.md
├── bin # Contains the crates that produce a binary or target the web  
├── cargo.toml # Set the workspace wide settings may contain some build scripts
├── models # Contains the lib crates that define a models
├── utils # Contains a crate that define useful struct, trait and functions 
└── notebooks # Groups the notebooks for test and visualization
```

## Install :

### Install libtorch :  

[For installing libtorch I live you to the pytorch documentation](https://pytorch.org/cppdocs/installing.html)

### Preping tch-rs :

Set your environement variables in your ``.bashrc`` to point the linker and the compilator to libtorch

```sh
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

#### CUDA :

To use cuda with your models, set ``TORCH_CUDA_VERSION`` to ``cu113``


## Models :

Here's a list of currently implemented models so far : 
- Multi Layer Perceptron 

## Experiments : 

### MNIST : 

Just a basic training on mnist to understand the basics of tch-rs