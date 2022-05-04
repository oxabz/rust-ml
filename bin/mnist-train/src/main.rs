use anyhow::Result;
use clap::{ArgEnum, Parser};
use mlp::MLP;
use tch::{
    nn::{self, Module, OptimizerConfig},
    Device,
};

#[derive(Debug, Parser)]
#[clap(version, author, about)]
struct Args {
    /// Path to the dataset
    mnist_path: String,

    /// Path to the save location for the weight of the model
    #[clap(long)]
    weight_path: Option<String>,

    /// Number of epoch runned
    #[clap(long, default_value_t = 500)]
    epoch: u32,

    /// Number of epoch runned
    #[clap(long, arg_enum, default_value_t = NormalizationParam::None)]
    normalization: NormalizationParam,

    /// Number of Nodes in the hiden layers
    #[clap(long, default_value_t = 128)]
    hidden_nodes: u32,

    /// Number of hidden layers
    #[clap(long, default_value_t = 2)]
    layer_count: u32,
}

#[derive(Debug, Copy, Clone, PartialEq, ArgEnum)]
enum NormalizationParam {
    None,
    Sigmoid,
}

fn main() -> Result<()> {
    // Parsing parameters
    let args = Args::parse();

    // Loading Dataset
    let m = tch::vision::mnist::load_dir(args.mnist_path)?;

    // Picking the device to use to the train of the model
    let device = if tch::Cuda::is_available() {
        let device_count = tch::Cuda::device_count() as f32;
        let r: f32 = rand::random();
        Device::Cuda(f32::floor(device_count * r) as usize)
    } else {
        Device::Cpu
    };

    // Creating the Model and the storage for the parameters
    let vs = nn::VarStore::new(device);
    let net = MLP::new(&vs.root(), 784, 10, args.hidden_nodes, args.layer_count);

    // Creating the optimizer
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    println!("{}", m.test_images.requires_grad());

    // Simple epoch loop
    for epoch in 1..args.epoch {
        // MNIST is small enought so that we do not have to split it in batch so we compute the loss directly on the whole dataset
        let y_hat = net.forward(&m.train_images.to_device(device));

        let y_hat = match args.normalization {
            NormalizationParam::None => y_hat,
            NormalizationParam::Sigmoid => y_hat.sigmoid(),
        };

        let loss = y_hat.cross_entropy_for_logits(&m.train_labels.to_device(device));

        // Gradient descent
        opt.backward_step(&loss);

        // Loggin the accuracy of the epoch
        let test_accuracy = net
            .forward(&m.test_images.to_device(device))
            .accuracy_for_logits(&m.test_labels.to_device(device));
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::from(&loss),
            100. * f64::from(&test_accuracy),
        );
    }

    if let Some(save_path) = args.weight_path {
        vs.save(save_path)?;
    }

    Ok(())
}
