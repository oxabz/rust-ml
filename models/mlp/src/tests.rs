#[cfg(test)]
mod tests {
    use tch::{nn::{self, Module}, Device, Tensor, Kind};
    use crate::MLP;

    #[test]
    fn create() {
        let vs = nn::VarStore::new(Device::Cpu);
        let _ = MLP::new(&(&vs.root() / "model"), 4, 3, 2, 1);
    }

    #[test]
    fn create_cuda() {
        tch::maybe_init_cuda();
        if tch::Cuda::is_available(){
            let vs = nn::VarStore::new(Device::Cuda(0));
            let _ = MLP::new(&(&vs.root() / "model"), 4, 3, 2, 1);
        }
        else {
            print!("Cuda unavailable")
        }
    }

    #[test]
    fn forwarding(){

        let vs = nn::VarStore::new(Device::Cpu);
        let mlp = MLP::new(&(&vs.root() / "model"), 4, 3, 5, 1);

        let x = Tensor::rand(&[2, 4], (Kind::Float, Device::Cpu));
        let y = mlp.forward(&x);
        
        assert_eq!(y.size(), vec![2, 3], "expected size [2,3] ");
    }

    #[test]
    fn forwarding_cuda(){

        let vs = nn::VarStore::new(Device::Cuda(0));
        let mlp = MLP::new(&(&vs.root() / "model"), 4, 3, 5, 1);

        let x = Tensor::rand(&[2, 4], (Kind::Float, Device::Cuda(0)));
        let y = mlp.forward(&x);
        
        assert_eq!(y.size(), vec![2, 3], "expected dim [2,3]");
    }
}
