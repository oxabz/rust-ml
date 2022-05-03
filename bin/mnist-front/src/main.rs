mod drawable_canvas;
mod inference_agent;

use crate::drawable_canvas::DrawableCanvas;
use crate::inference_agent::{InferenceAgent, InferenceAgentInput, InferenceAgentOutput};

use log::info;
use yew::prelude::*;
use yew_agent::use_bridge;

#[function_component(Input)]
fn input() -> Html{
    let inference_agent = use_bridge::<InferenceAgent,_>(|_|{});
    let onchange = {
        let inference_agent = inference_agent.clone();
        Callback::from(move|data| inference_agent.send(InferenceAgentInput::NewInput(data)))
    };
    html! { 
        <div class="container ">
            <h2>{"Input"}</h2>
            <div class="hcenter">
                <DrawableCanvas width={28} height={28} id="input-canvas" canvasid="input-canvas-inner" class="article" {onchange}
                    color={"white"} background={"black"}/>
            </div>
        </div>
    }
}

#[function_component(Output)]
fn output() -> Html{
    let values = use_state(||InferenceAgentOutput::Unitialized);
    let _inference_agent = {
        let values = values.clone();
        use_bridge::<InferenceAgent,_>(move |out|{info!("change");values.set(out)})
    };
    let res = match &*values {
        InferenceAgentOutput::Clean(val) | InferenceAgentOutput::Dirty(val) => {
            let headers = (0..=9).map(|i:i32|html!{<th>{i}</th>});
            let num_val = val.iter().map(|v|html!{<th>{format!("{v:.2}")}</th>});
            html!{
                <figure>
                <table>
                    <thead>
                        <tr>
                            {for headers}
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            {for num_val}
                        </tr>
                    </tbody>
                </table>
                </figure>
            }
        },
        InferenceAgentOutput::Unitialized => html!{},
    };
    html! { 
        <div class="container">
            <h2>{"Output"}</h2>
            <article>
                {res}
            </article>
        </div>
    }
}

#[function_component(App)]
fn app() -> Html {
    html! { 
        <>
            <header class="container">
                <hgroup>
                    <h1>{"Inference for MNIST"}</h1>
                    <h2>{"Small interface for an API trained on MNIST."}</h2>
                </hgroup>
            </header>
            <div class="container">
                <Input/>
                <Output/>
            </div>
        </>
    }
}

fn main() {
    wasm_logger::init(wasm_logger::Config::default());
    yew::start_app::<App>();
}
