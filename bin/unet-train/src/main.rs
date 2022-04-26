use std::{path::{PathBuf}, fs::File};
use clap::{Parser, ArgEnum};
use itertools::{Itertools, multiunzip};
use tch::{Tensor, nn::{self, OptimizerConfig, Module}, Device, vision::image};
use tch_utils::{data::{Datafolder}, metrics::dice_score_1c};
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
        let mut steps = 0;
        let mut avg_loss = 0.0;
        let train_ds = Datafolder::from(&train_path, "images".to_string(), "mask".to_string(), &load_image, &|path|tch::vision::image::load(path).unwrap())?
            .map(|x|{
                println!("loaded");
                (image::resize(&x.0, 565, 565).unwrap(),image::resize(&x.0, 565, 565).unwrap())
            });
        for batch in train_ds.chunks(args.batch_size).into_iter(){
            let (x, y) : (Vec<_>,Vec<_>) = multiunzip(batch);
            let x = Tensor::stack(x.as_slice(), 0).to_kind(tch::Kind::Float).to_device(device);
            let y = Tensor::stack(y.as_slice(), 0).to_kind(tch::Kind::Float).to_device(device);
            // Making the prediction
            let y_hat = unet.forward(&x);
            let loss = 1.0 as f32 - dice_score_1c(&y_hat, &y);
            
            // Gradient descent
            opt.backward_step(&loss);

            steps += args.batch_size;
            avg_loss += (f64::from(&loss) - avg_loss) / (steps + args.batch_size) as f64;
        }
        
        // Loggin the loss of the epoch
        println!(
            "epoch: {:4} train loss: {:8.5}",
            epoch,
            avg_loss
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