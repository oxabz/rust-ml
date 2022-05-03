
use std::collections::HashSet;

use log::error;
use yew_agent::{Agent, Context, AgentLink, HandlerId};

const API_URL: &'static str= "API/V1/";

#[derive(Clone)]
pub enum InferenceAgentOutput{
    Clean(Vec<f64>),
    Dirty(Vec<f64>),
    Unitialized
}

pub enum InferenceAgentInput{
    NewInput(Vec<u8>)
}

pub enum InferenceAgentMessage{
    ResultRecieved(Vec<f64>),
    RequestError
}

pub struct InferenceAgent{
    link:AgentLink<Self>,
    url: String,
    result: InferenceAgentOutput,
    subscribers: HashSet<HandlerId>
}

impl InferenceAgent{
    async fn request_inference(url:String, data:Vec<u8>) -> InferenceAgentMessage{
        let mut resp = match surf::post(&url).body(data).await{
            Ok(ok) => ok,
            Err(err) => {
                error!("InferenceAgent::request_inference {err}");
                return InferenceAgentMessage::RequestError;
            }
        };

        let res = match resp.body_json::<Vec<f64>>().await{
            Ok(res)=>res,
            Err(err) => {
                error!("InferenceAgent::request_inference {err}");
                return InferenceAgentMessage::RequestError;
            }
        };

        InferenceAgentMessage::ResultRecieved(res) 
    }
}

impl Agent for InferenceAgent{
    type Reach = Context<InferenceAgent>;
    type Message = InferenceAgentMessage;
    type Input = InferenceAgentInput;
    type Output = InferenceAgentOutput;

    fn create(link: AgentLink<Self>) -> Self {
        let window = web_sys::window().unwrap();
        let base_url = window.location().origin().unwrap();
        let url = format!("{base_url}/{API_URL}");
        Self{
            link,
            result:InferenceAgentOutput::Unitialized,
            url,
            subscribers: HashSet::new(), 
        }
    }

    fn update(&mut self, msg: Self::Message) {
        match msg{
            InferenceAgentMessage::ResultRecieved(res) => {
                self.result = InferenceAgentOutput::Clean(res);
                self.subscribers.iter().for_each(|s|self.link.respond(s.clone(), self.result.clone()))
            },
            InferenceAgentMessage::RequestError => {},
        }
    }

    fn handle_input(&mut self, msg: Self::Input, _id: yew_agent::HandlerId) {
        match msg{
            InferenceAgentInput::NewInput(data) => {
                if let InferenceAgentOutput::Clean(v) = self.result.clone(){
                    self.result = InferenceAgentOutput::Dirty(v)
                }
                let url = self.url.clone();
                self.link.send_future(Self::request_inference(url, data));
            },
        }
    }

    fn connected(&mut self, id: HandlerId) {
        self.subscribers.insert(id);
    }

    fn disconnected(&mut self, id: HandlerId) {
        self.subscribers.remove(&id);
    }
    
}