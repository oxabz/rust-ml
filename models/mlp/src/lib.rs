mod tests;

use tch::{nn, Tensor};

#[derive(Debug)]
pub struct MLP(nn::Sequential); 

impl MLP{
    pub fn new(vs: &nn::Path, in_feature:u32, out_feature:u32, hidden_nodes: u32, layer_count:u32)-> Self{
        let seq = nn::seq();
        let in_layer = nn::linear(
            vs / "layer0", 
            in_feature as i64,
            hidden_nodes as i64, 
            Default::default()
        );
    
        assert!(in_feature>0, "in_features should be above 0");
        assert!(out_feature>0, "out_features should be above 0");
        assert!(hidden_nodes>0, "hidden_nodes should be above 0");
        assert!(layer_count>0, "layer_count should be above 0");
    
        let seq = seq.add(in_layer);
        let seq = seq.add_fn(Tensor::relu);
    
        let inter_layers = (1..layer_count).into_iter().map(|i| nn::linear(vs / format!("layer{}", i), hidden_nodes as i64, hidden_nodes as i64, Default::default()));
    
        let seq = inter_layers.fold(seq, |seq, lin| seq.add(lin).add_fn(Tensor::relu));
    
        let out_layer = nn::linear(
            vs / format!("layer{}", layer_count), 
            hidden_nodes as i64,
            out_feature as i64, 
            Default::default()
        );
        MLP(seq.add(out_layer))
    }
}

impl nn::Module for MLP {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.0.forward(xs)
    }
}