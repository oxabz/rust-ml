use tch::{
    nn::{self, ConvConfig, ConvTranspose2D, ConvTransposeConfig, Module, Path, Sequential},
    Tensor,
};

pub struct VruNet<E: tch_utils::types::FeatureExtractor<L>, const L: usize> {
    encoder: E,
    center: Sequential,
    decoder: Vec<(ConvTranspose2D, Sequential)>,
}

impl<E: tch_utils::types::FeatureExtractor<L>, const L: usize> VruNet<E, L> {
    pub fn new(vs: &Path, encoder: E) -> Self {
        let conf = ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let center = {
            let center = nn::seq();
            let vs = vs / "center";
            center
                .add(nn::conv2d(
                    &(&vs / "conv1"),
                    E::CHANELS_COUNT[L - 1],
                    E::CHANELS_COUNT[L - 1] * 2,
                    3,
                    conf,
                ))
                .add_fn(Tensor::relu)
                .add(nn::conv2d(
                    &(&vs / "conv1"),
                    E::CHANELS_COUNT[L - 1] * 2,
                    E::CHANELS_COUNT[L - 1] * 2,
                    3,
                    conf,
                ))
                .add_fn(Tensor::relu)
        };
        let conft = ConvTransposeConfig {
            stride: 2,
            ..Default::default()
        };
        dbg!(&conf);
        let decoder = {
            let vs = vs / "decoder";
            E::CHANELS_COUNT
                .iter()
                .enumerate()
                .rev()
                .map(|(l, fc)| {
                    let vs = &vs / format!("layer{l}");
                    let sub_sampl_dim = if l == L - 1 { 4 } else { 0 };
                    let up = nn::conv_transpose2d(&(&vs / "upconv"), fc * 2, *fc, 2, conft);
                    let convs = nn::seq()
                        .add(nn::conv2d(
                            &(&vs / "conv1"),
                            fc * 2 + sub_sampl_dim,
                            *fc,
                            3,
                            conf,
                        ))
                        .add_fn(Tensor::relu)
                        .add(nn::conv2d(&(&vs / "conv2"), *fc, *fc, 3, conf))
                        .add_fn(Tensor::relu);
                    (up, convs)
                })
                .collect()
        };
        Self {
            encoder,
            center,
            decoder,
        }
    }

    pub fn forward(&self, xp1: &Tensor, xp2: &Tensor) -> Tensor {
        // NOTE : The specification of the decoder is bad returning an array doesnt allow to drop progressively the tensor as they are not needed
        let (fms, bot) = self.encoder.forward_extracts(xp1);

        let c = {
            let bot = bot; // We move tensors so that they can quickly be droped
            self.center.forward(&bot)
        };

        let mut dec = self.decoder.iter();

        let x = {
            let c = c;
            let (up, convs) = dec
                .next()
                .expect("The network should at least have one layer");

            let x = up.forward(&c);
            let x = Tensor::cat(&[&x, &fms[L - 1], xp2], -3);
            convs.forward(&x)
        };

        dec.enumerate().fold(x, |x, (i, (up, convs))| {
            let x = up.forward(&x);
            let x = Tensor::cat(&[&x, &fms[L - i - 2]], -3);
            convs.forward(&x)
        })
    }
}

#[cfg(test)]
mod tests {
    use tch::{
        nn::{ConvConfig, VarStore},
        Device, Kind, Tensor,
    };
    use unet::encoder::BasicCNN;

    use crate::VruNet;

    #[test]
    fn it_works() {
        let xa = Tensor::rand(&[10, 7, 256, 256], (Kind::Float, Device::Cpu));
        let xb = Tensor::rand(&[10, 4, 32, 32], (Kind::Float, Device::Cpu));

        print!("{}", xa.requires_grad());
        print!("{}", xb.requires_grad());

        let vs = VarStore::new(Device::Cpu);

        let cnv_cfg = ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let encoder = BasicCNN::new_conf(&(&vs.root() / "encoder"), 7, cnv_cfg);
        let tnet = VruNet::new(&vs.root(), encoder);

        let _res = tnet.forward(&xa, &xb);
    }
}
