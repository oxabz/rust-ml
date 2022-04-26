use tch::{nn::{self, Path, Sequential, Module}};
use tch_utils::types::FeatureExtractor;



#[derive(Debug)]
pub struct BasicCNN{
    layers: Vec<Sequential>
}

impl BasicCNN {
    pub fn new(vs:&Path, in_channels: u32)-> Self{
        let mut previous_channels = in_channels as i64;    
        let mut layers = vec![];
        for i in 0..4{
            let seq = nn::seq();
            let seq = seq.add(nn::conv2d(&(vs/format!("layer{i}"))/"0", previous_channels, 64*(2 as i64).pow(i), 3, Default::default()))
                .add(nn::conv2d(&(vs/format!("layer{i}"))/"1", 64*(2 as i64).pow(i), 64*(2 as i64).pow(i), 3, Default::default()));
            previous_channels = 64*(2 as i64).pow(i);
            layers.push(seq);
        }
        Self{
            layers
        }
    } 
}

impl nn::Module for BasicCNN{
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        let xs = self.layers[0].forward(xs).max_pool2d_default(2);
        let xs = self.layers[1].forward(&xs).max_pool2d_default(2);
        let xs = self.layers[2].forward(&xs).max_pool2d_default(2);
        let xs = self.layers[3].forward(&xs).max_pool2d_default(2);
        xs
    }
}

impl FeatureExtractor<4> for BasicCNN{
    const CHANELS_COUNT: [i64; 4] = [64, 128, 256, 512];

    fn forward_extracts(&self, xs: &tch::Tensor) -> ([tch::Tensor; 4], tch::Tensor) {
        let x0 = self.layers[0].forward(xs);
        let x1 = self.layers[1].forward(&x0.max_pool2d_default(2));
        let x2 = self.layers[2].forward(&x1.max_pool2d_default(2));
        let x3 = self.layers[3].forward(&x2.max_pool2d_default(2));
        let y = x3.max_pool2d_default(2);
        ([x0, x1, x2, x3], y)
    }
}