use async_channel::{unbounded, Receiver};
use bytes::Bytes;
use tch::{nn::{self, Module}, Kind, Tensor, IndexOp};
use tokio::sync::oneshot::{channel, Sender};
use warp::{Filter, reply::with_status, hyper::StatusCode, Reply, Rejection, path};
use clap::Parser;


#[derive(Debug, Parser)]
#[clap(version, author, about)]
struct Args{
    /// Number of workers used for inference
    #[clap(short, long, default_value_t = 4)]
    workers: u32,
    /// Port to listen on 
    #[clap(short, long, default_value_t = 3030)]
    port: u16,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 10)]
async fn main() {
    let args = Args::parse();

    // Creating  Worker threads that will handle the inference along side with a queue chanel to send them the input (we keep an instance of the sender in the main thread to keep the worker alive.)
    let (s, r) = unbounded::<(Tensor, Sender<Vec<f64>>)>();
    for _ in 0..args.workers {
        let r = r.clone();
        tokio::spawn(async {inference_worker(r).await});
    }

    // Setting up a route to do the inference 
    // GET /
    let mnist_inf_route = path!("API"/"V1")
        .and(warp::post())
        .map(move||s.clone()) // Requires the sender to ask for inference to the worker 
        .and(warp::body::bytes()) // Need a form to recieve the image
        .and_then(inference_controller); 

    // Start the server
    warp::serve(mnist_inf_route.with(warp::filters::log::log("requests")))
        .run(([0, 0, 0, 0], args.port))
        .await;
}

async fn inference_worker(tasks: Receiver<(Tensor, Sender<Vec<f64>>)>){
    // Setting up an instance of the model for the worker
    let mut vs = nn::VarStore::new(tch::Device::Cpu);
    let mlp = mlp::MLP::new(&vs.root(), 784, 10, 128, 2);
    vs.load("weights.pt").expect("unable to load weights");

    // Recieve a inference request, compute the result and send it back with the given one shot chanel
    while let Ok((x, s)) = tasks.recv().await {
        let y_hat = mlp.forward(&x.reshape(&[784])).softmax(0, Kind::Float);
        let res = (0..=9).into_iter().map(|i|f64::from(y_hat.i(i))).collect::<Vec<_>>();
        if let Err(err) = s.send(res){
            eprintln!("Couldnt respond to the client : {err:?}");
        }
    } 
}

async fn inference_controller( queue:async_channel::Sender<(Tensor, Sender<Vec<f64>>)>, data: Bytes)->Result<impl Reply, Rejection> {
    let data:Vec<u8> = data.into_iter().collect();

    let img = match tch::Tensor::try_from(data){
        Ok(ok) => ok,
        Err(_) => return Err(warp::reject()),
    };

    if img.size1().unwrap() != 3136{
        return Ok(with_status(format!("Expected 28x28x4 image"), StatusCode::BAD_REQUEST));
    }

    let img = img.reshape(&[28, 28, 4]).swapaxes(0, 2).swapaxes(1, 2);
    
    // Checking the dimention of the tensor 
    if img.size()[1] != 28 || img.size()[2] != 28 {
        return Ok(with_status("Expected 28x28 images".to_string(), StatusCode::BAD_REQUEST))
    }

    // Preparing the tensor for inference (float type & gray scale)
    let img = img.to_kind(Kind::Float).i((0,..,..));

    // Sending the input to be processed by the worker
    let (sret, rret) = channel::<Vec<f64>>();
    if queue.send((img, sret)).await.is_err() {
        return Ok(with_status("Weird chanel stuff".to_string(), StatusCode::INTERNAL_SERVER_ERROR));
    }

    // Awaiting the answer of the worker on the oneshot chanel
    let res = match rret.await {
        Ok(res) => res,
        Err(_) => return Ok(with_status("Weird chanel stuff".to_string(), StatusCode::INTERNAL_SERVER_ERROR))
    };

    // HTTP Reply
    Ok(with_status(format!("{res:?}"), StatusCode::OK))
}