use tch::{nn::{Sequential, Conv2D, Path,self, ConvTranspose2D}, Tensor, index::IndexOp};
use tch_utils::types::FeatureExtractor;
pub mod encoder;

pub struct UnetProps{
    pub decoder_block_convolutions: u32,
    pub center_block_convolutions: u32,
    pub input_chanels: i64,
}

#[derive(Debug)]
pub struct UNet<E, const L : usize>
where E: FeatureExtractor<L>{
    encoder: E,
    center: Sequential,
    decoder: Vec<(usize, ConvTranspose2D, Sequential)>,
    classifier: Conv2D
}

impl<E,const L : usize> UNet<E, L> where E:FeatureExtractor<L>{
    pub fn new(vs:&Path, encoder:E, class_count:u32, props: UnetProps)->Self{
        let layer_count = L;

        let conv_conf = nn::ConvConfig{
            ..Default::default()
        };

        let chanels = <E as FeatureExtractor<L>>::CHANELS_COUNT;

        // Creating the center convolutions
        let center = nn::seq();
        let center = center.add(nn::conv2d(&(vs / "center") / format!("conv0"), chanels[layer_count-1], chanels[layer_count-1]*2, 3, conv_conf)).add_fn(Tensor::relu);
        let center = (1..props.center_block_convolutions).into_iter()
            .fold(center, |seq, i| {
                let conv = nn::conv2d(&(vs / "center") / format!("conv{i}"), chanels[layer_count-1]*2, chanels[layer_count-1]*2, 3, conv_conf);
                seq.add(conv).add_fn(Tensor::relu)
            });

        // Creating the decoder layers
        let decoder_vs = &(vs / "decoder");
        let decoder: Vec<_> = (0..layer_count).rev().into_iter()
            .map(|layer|{
                let vs = decoder_vs/ format!("layer{layer}");
                let seq = nn::seq();

                // Creating the up-convolution and the first convolution
                let upconv = nn::conv_transpose2d(&vs / "upconv", chanels[layer]*2, chanels[layer], 2, nn::ConvTransposeConfigND { stride: 2, padding: 0, ..Default::default()});
                let seq = seq.add(nn::conv2d(&vs / "conv0", chanels[layer]*2, chanels[layer], 3, conv_conf)).add_fn(Tensor::relu); 
                
                // Adding the others convolutions
                let seq = (1..props.decoder_block_convolutions).into_iter()
                    .fold(seq, |seq, i| {
                        let conv = nn::conv2d(&vs / format!("conv{i}"), chanels[layer_count-1]*2, chanels[layer_count-1]*2, 3, conv_conf);
                        seq.add(conv).add_fn(Tensor::relu)
                    });
                (layer, upconv, seq)
            }).collect();
            

        // Last convolution to classify pixels
        let classifier = nn::conv2d(&(vs/"classifier"), chanels[0], class_count as i64, 1, conv_conf);

        Self{
            encoder,
            center,
            decoder,
            classifier,
        }
    }

}

impl<E,const L : usize> nn::Module for UNet<E, L> where E:FeatureExtractor<L> {
    fn forward(&self, xs: &Tensor) -> Tensor {
        assert!(xs.dim()==3||xs.dim()==4, "Expected [C,W,H]/[B,C,W,H] shaped tensor got {:?} instead", xs.size());

        // Extracting features with the encoder
        let (feature_maps, xs) = <E as FeatureExtractor<L>>::forward_extracts(&self.encoder, xs);

        // Taking the last feature map to be processed by the center convolutions 
        let mut xs = self.center.forward(&xs);
        
        for (layer, upconv, convs) in self.decoder.iter(){
            // Upsampling
            let xt = upconv.forward(&xs);
            let fm = &feature_maps[*layer];

            //Cropping the bypass
            let x_size = xt.size();
            let fm_size = fm.size();
            let dw = fm_size[fm_size.len()-2] - x_size[fm_size.len()-2];
            let dh = fm_size[fm_size.len()-1] - x_size[x_size.len()-1];
            let resized_fm = if fm_size.len() == 3{
                fm.i((.., dw/2..fm_size[fm_size.len()-2]-dw/2, dh/2..fm_size[fm_size.len()-1]-dh/2))
            } else {
                fm.i((.., .., dw/2..fm_size[fm_size.len()-2]-dw/2, dh/2..fm_size[fm_size.len()-1]-dh/2))
            };
            let stacked = tch::Tensor::stack(&[xt, resized_fm], (fm_size.len()-3) as i64);
            
            //Convolutions
            xs = convs.forward(&stacked);
        }
        return self.classifier.forward(&xs);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
