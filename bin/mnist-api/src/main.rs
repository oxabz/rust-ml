use tch::{nn::{self, Module}, Kind};
use warp::{Filter, multipart::{FormData, Part}, reply::with_status, hyper::StatusCode, Reply, Rejection, Buf};
use futures_util::TryStreamExt;

#[tokio::main]
async fn main() {
    let mnist_inf_route = warp::get()
        .and(warp::multipart::form())
        .and_then(inference);

    // GET /hello/warp => 200 OK with body "Hello, warp!"
    let hello = warp::path!("hello" / String)
        .map(|name| format!("Hello, {}!", name));

    warp::serve(hello.or(mnist_inf_route))
        .run(([0, 0, 0, 0], 3030))
        .await;
}


async fn inference(form: FormData)->Result<impl Reply, Rejection> {
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

    let mut vs = nn::VarStore::new(tch::Device::Cpu);
    let mlp = mlp::MLP::new(&vs.root(), 784, 10, 128, 2);
    vs.load("weights.pt");

    let y_hat = mlp.forward(&img.to_kind(Kind::Float).mean_dim(&[0], false,Kind::Float).reshape(&[784])).argmax( None,false);
    dbg!(&y_hat);
    Ok(with_status(format!("{}", i32::from(&y_hat)), StatusCode::OK))
}