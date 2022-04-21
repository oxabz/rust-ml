# mnist-train 

A test to understand the basics of tch-rs 

## How to run it ?

- [Download mnist](http://yann.lecun.com/exdb/mnist/)
- extract files 
- mv them into folder with the following names :
```
data
├── t10k-images-idx3-ubyte
├── t10k-labels-idx1-ubyte
├── train-images-idx3-ubyte
└── train-labels-idx1-ubyte
```
- cd to the root of the crate 
- ``cargo run -- path/to/mnist``

## How to save the weights
``cargo run -- path/to/mnist --weight_path path/to/weights``

## Usage
