use anyhow::Result;
use tch::{Device, nn::{self, OptimizerConfig, Module}};
use mlp::MLP;

fn main() -> Result<()> {
    // Loading Dataset
    let m = tch::vision::mnist::load_dir("data")?;
    let vs = nn::VarStore::new(Device::Cuda(0));
    let net = MLP::new(&vs.root(), 784, 10, 128, 2);
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 1..500 {
        let loss = net
            .forward(&m.train_images.to_device(Device::Cuda(0)))
            .cross_entropy_for_logits(&m.train_labels.to_device(Device::Cuda(0)));
        opt.backward_step(&loss);
        let test_accuracy = net
            .forward(&m.test_images.to_device(Device::Cuda(0)))
            .accuracy_for_logits(&m.test_labels.to_device(Device::Cuda(0)));
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::from(&loss),
            100. * f64::from(&test_accuracy),
        );
    }
    Ok(())
}
