use std::{path::{PathBuf, Path}, fs::File};
use clap::{Parser, ArgEnum};
use itertools::{Itertools, multiunzip};
use tch::{Tensor, vision::image, nn::{self, OptimizerConfig, Module}, Device};
use tch_utils::data::{Dataset, Datafolder};
use tiff::decoder::Decoder;
use unet::encoder;


#[derive(Debug, Copy, Clone, PartialEq, ArgEnum)]
enum SuportedEncoders {
    BasicCNN
}

#[derive(Debug, Parser)]
#[clap(version, author, about)]
struct Args{
    /// Path to the dataset expect the folder to contain two folder train & test with each a images & mask folder 
    dataset_path: PathBuf,

    /// Path to the save location for the weight of the model
    #[clap(long)]
    weight_path: Option<String>,
    #[clap(long, default_value_t = 16)]
    batch_size: usize,
    
    /// Encoder used in the U-Net
    #[clap(arg_enum)]
    encoder: SuportedEncoders
}

fn main() -> anyhow::Result<()>{
    let args = Args::parse();

    let train_path = args.dataset_path.join("training");
    let mut train_ds = Datafolder::from(&train_path, "images".to_string(), "mask".to_string(), &load_image, &|path|tch::vision::image::load(path).unwrap())?;

    let (x, y) = train_ds.next().unwrap();
    image::save(&x, "test.png");
    
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
    let encoder = encoder::BasicCNN::new(&(&vs.root() / "encoder"), 3);
    let unet = unet::UNet::new(&vs.root(), encoder, 1, Default::default());
    
    // Creating the optimizer 
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    // Simple epoch loop
    for epoch in 1..500 {
        for batch in train_ds.chunks(args.batch_size).into_iter(){
            let (x, y) : (Vec<_>,Vec<_>) = multiunzip(batch);
            let x = Tensor::stack(x.as_slice(), 0);
            let y = Tensor::stack(y.as_slice(), 0);
            // MNIST is small enought so that we do not have to split it in batch so we compute the loss directly on the whole dataset
            let loss = unet
                .forward(&x.train_images.to_device(device.clone()));
                //.(&m.train_labels.to_device(device.clone()));
                
            // Gradient descent
            opt.backward_step(&loss);

        }
        
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

fn load_image(path:PathBuf)-> Tensor{
    let file = File::open(path).unwrap();
    let mut decoder = Decoder::new(file).unwrap();
    let (x,y) = decoder.dimensions().unwrap();
    let image = decoder.read_image().unwrap();
    
    let img = match image {
        tiff::decoder::DecodingResult::U8(i) => i,
        _=>panic!()
    };
    Tensor::from(img.as_slice()).reshape(&[ y as i64, x as i64, 3]).swapaxes(0, 2).swapaxes(1, 2)
}