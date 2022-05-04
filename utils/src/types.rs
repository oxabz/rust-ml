pub trait FeatureExtractor<const L: usize>: tch::nn::Module {
    const CHANELS_COUNT: [i64; L];
    fn forward_extracts(&self, xs: &tch::Tensor) -> ([tch::Tensor; L], tch::Tensor);
}
