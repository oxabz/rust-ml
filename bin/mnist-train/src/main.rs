use std::f32;

use anyhow::Result;
use tch::{Device, nn::{self, OptimizerConfig, Module}};
use mlp::MLP;

fn main() -> Result<()> {
    // Loading Dataset
    let m = tch::vision::mnist::load_dir("data")?;

    // Picking the device to use to the train of the model
    let device = if tch::Cuda::is_available(){
        let device_count = tch::Cuda::device_count() as f32;
        let r : f32 = rand::random();
        Device::Cuda(f32::floor(device_count*r) as usize)
    } else {
        Device::Cpu
    };

    // Creating the Model and the storage for the parameters
    let vs = nn::VarStore::new(device.clone());
    let net = MLP::new(&vs.root(), 784, 10, 128, 2);

    // Creating the optimizer 
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    // Simple epoch loop
    for epoch in 1..500 {
        
        // MNIST is small enought so that we do not have to split it in batch so we compute the loss directly on the whole dataset
        let loss = net
            .forward(&m.train_images.to_device(device.clone()))
            .cross_entropy_for_logits(&m.train_labels.to_device(device.clone()));
        
        // Gradient descent
        opt.backward_step(&loss);

        // Loggin the accuracy of the epoch
        let test_accuracy = net
            .forward(&m.test_images.to_device(device.clone()))
            .accuracy_for_logits(&m.test_labels.to_device(device.clone()));
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::from(&loss),
            100. * f64::from(&test_accuracy),
        );
    }
    
    Ok(()) 
}
