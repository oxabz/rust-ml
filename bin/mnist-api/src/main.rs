use async_channel::{unbounded, Receiver};
use tch::{nn::{self, Module}, Kind, Tensor};
use tokio::sync::oneshot::{channel, Sender};
use warp::{Filter, multipart::{FormData, Part}, reply::with_status, hyper::StatusCode, Reply, Rejection, Buf};
use futures_util::TryStreamExt;



#[tokio::main]
async fn main() {

    let (s, r) = unbounded::<(Tensor, Sender<i32>)>();

    for _ in 0..1 {
        let r = r.clone();
        tokio::spawn(async {model_worker(r).await});
    }

    let mnist_inf_route = warp::get()
        .map(move||s.clone())
        .and(warp::multipart::form())
        .and_then(inference);

    warp::serve(mnist_inf_route.with(warp::filters::log::log("")))
        .run(([0, 0, 0, 0], 3030))
        .await;
}

async fn model_worker(tasks: Receiver<(Tensor, Sender<i32>)>){
    let mut vs = nn::VarStore::new(tch::Device::Cpu);
    let mlp = mlp::MLP::new(&vs.root(), 784, 10, 128, 2);
    vs.load("weights.pt").expect("unable to load weights");

    while let Ok((x, s)) = tasks.recv().await {
        let y_hat = mlp.forward(&x.to_kind(Kind::Float).mean_dim(&[0], false,Kind::Float).reshape(&[784])).argmax( None,false);
        if let Err(err) = s.send(i32::from(&y_hat)){
            eprintln!("Couldnt respond to the client {err:?}");
        }
    } 
}

async fn inference( queue:async_channel::Sender<(Tensor, Sender<i32>)>, form: FormData)->Result<impl Reply, Rejection> {
    let parts: Vec<Part> = form.try_collect().await.map_err(|e| {
        eprintln!("form error: {}", e);
        warp::reject::reject()
    })?;

    let mut file_part  = match parts.into_iter().find(|x|x.name()=="file") {
        Some(p)=>p,
        None => return Ok(with_status("Need a file for inference".to_string(), StatusCode::BAD_REQUEST))
    };

    let content_type = match file_part.content_type() {
        Some(t) => Ok(t),
        None => Err(warp::reject::reject()),
    }?;

    if content_type != "image/png" {
        return Ok(with_status("Only accept png".to_string(), StatusCode::BAD_REQUEST)) 
    }

    let t = file_part.data().await.unwrap().unwrap();
    let img = match tch::vision::image::load_from_memory(t.chunk()){
        Ok(img) => img,
        Err(_) => return Ok(with_status("error loading the image".to_string(), StatusCode::BAD_REQUEST)),
    };

    if img.size()[1] != 28 || img.size()[2] != 28 {
        return Ok(with_status("Expected 28x28 images".to_string(), StatusCode::BAD_REQUEST))
    }

    let (sret, rret) = channel::<i32>();
    if queue.send((img, sret)).await.is_err() {
        return Ok(with_status("Weird chanel stuff".to_string(), StatusCode::INTERNAL_SERVER_ERROR));
    }

    let res = match rret.await {
        Ok(res) => res,
        Err(_) => return Ok(with_status("Weird chanel stuff".to_string(), StatusCode::INTERNAL_SERVER_ERROR))
    };

    Ok(with_status(format!("{}", res), StatusCode::OK))
}