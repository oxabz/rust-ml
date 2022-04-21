use async_channel::{unbounded, Receiver};
use tch::{nn::{self, Module}, Kind, Tensor};
use tokio::sync::oneshot::{channel, Sender};
use warp::{Filter, multipart::{FormData, Part}, reply::with_status, hyper::StatusCode, Reply, Rejection, Buf};
use futures_util::TryStreamExt;
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
    let (s, r) = unbounded::<(Tensor, Sender<i32>)>();
    for _ in 0..args.workers {
        let r = r.clone();
        tokio::spawn(async {inference_worker(r).await});
    }

    // Setting up a route to do the inference 
    // GET /
    let mnist_inf_route = warp::get()
        .map(move||s.clone()) // Requires the sender to ask for inference to the worker 
        .and(warp::multipart::form()) // Need a form to recieve the image
        .and_then(inference_controller); 

    // Start the server
    warp::serve(mnist_inf_route.with(warp::filters::log::log("requests")))
        .run(([0, 0, 0, 0], args.port))
        .await;
}

async fn inference_worker(tasks: Receiver<(Tensor, Sender<i32>)>){
    // Setting up an instance of the model for the worker
    let mut vs = nn::VarStore::new(tch::Device::Cpu);
    let mlp = mlp::MLP::new(&vs.root(), 784, 10, 128, 2);
    vs.load("weights.pt").expect("unable to load weights");

    // Recieve a inference request, compute the result and send it back with the given one shot chanel
    while let Ok((x, s)) = tasks.recv().await {
        let y_hat = mlp.forward(&x.reshape(&[784])).argmax( None,false);
        if let Err(err) = s.send(i32::from(&y_hat)){
            eprintln!("Couldnt respond to the client : {err}");
        }
    } 
}

async fn inference_controller( queue:async_channel::Sender<(Tensor, Sender<i32>)>, form: FormData)->Result<impl Reply, Rejection> {
    // Awaiting the complete form and collecting it in a colection of it's field
    let parts: Vec<Part> = form.try_collect().await.map_err(|e| {
        eprintln!("Form error: {}", e);
        warp::reject::reject()
    })?;

    // Picking the first file field
    let mut file_part  = match parts.into_iter().find(|x|x.name()=="file") {
        Some(p)=>p,
        None => return Ok(with_status("Need a file for inference".to_string(), StatusCode::BAD_REQUEST))
    };

    // Checking the type of the file to be a png
    let content_type = match file_part.content_type() {
        Some(t) => Ok(t),
        None => Err(warp::reject::reject()),
    }?;
    if content_type != "image/png" {
        return Ok(with_status("Only accept png".to_string(), StatusCode::BAD_REQUEST)) 
    }

    // Collecting the file data into a tensor
    let t = file_part.data().await.unwrap().unwrap();
    let img = match tch::vision::image::load_from_memory(t.chunk()){
        Ok(img) => img,
        Err(_) => return Ok(with_status("error loading the image".to_string(), StatusCode::BAD_REQUEST)),
    };

    // Checking the dimention of the tensor 
    if img.size()[1] != 28 || img.size()[2] != 28 {
        return Ok(with_status("Expected 28x28 images".to_string(), StatusCode::BAD_REQUEST))
    }

    // Preparing the tensor for inference (float type & gray scale)
    let img = img.to_kind(Kind::Float).mean_dim(&[0], false,Kind::Float);

    // Sending the input to be processed by the worker
    let (sret, rret) = channel::<i32>();
    if queue.send((img, sret)).await.is_err() {
        return Ok(with_status("Weird chanel stuff".to_string(), StatusCode::INTERNAL_SERVER_ERROR));
    }

    // Awaiting the answer of the worker on the oneshot chanel
    let res = match rret.await {
        Ok(res) => res,
        Err(_) => return Ok(with_status("Weird chanel stuff".to_string(), StatusCode::INTERNAL_SERVER_ERROR))
    };

    // HTTP Reply
    Ok(with_status(format!("{}", res), StatusCode::OK))
}