use log::{error, warn};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};
use yew::prelude::*;

const PAINT_SIZE: f64 = 1.5;

#[derive(Debug, Properties, PartialEq, Clone)]
pub struct DrawableCanvasProp {
    pub canvasid: Option<String>,
    pub id: Option<String>,
    pub class: String,
    pub width: usize,
    pub height: usize,
    pub onchange: Option<Callback<Vec<u8>>>,
    pub color: String,
    pub background: String,
}
pub struct DrawableCanvas {
    canvas_ref: NodeRef,
    pressed: bool,
    width: f64,
    height: f64,
    draw_context: Option<CanvasRenderingContext2d>,
    canvas: Option<HtmlCanvasElement>,
    color: String,
    background: String,
}

pub enum DrawableCanvasMessage {
    MouseDown(MouseEvent),
    MouseUp(MouseEvent),
    MouseTick(MouseEvent),
    ClearCanvas,
}

impl DrawableCanvas {
    fn draw_on_canvas(&self, offset_x: i32, offset_y: i32) {
        let canvas = match &self.canvas {
            Some(canvas) => canvas,
            None => {
                error!("DrawableCanvas::draw_on_canvas : Canvas hasnt been initiated!");
                return;
            }
        };
        let context = match &self.draw_context {
            Some(ctx) => ctx,
            None => {
                error!("DrawableCanvas::draw_on_canvas : Context hasnt been initiated!");
                return;
            }
        };

        let offset_width = canvas.offset_width();
        let offset_height = canvas.offset_height();

        let canvas_x = offset_x as f64 / offset_width as f64 * canvas.width() as f64;
        let canvas_y = offset_y as f64 / offset_height as f64 * canvas.width() as f64;

        context.set_fill_style(&JsValue::from(&self.color));
        context.fill_rect(
            canvas_x - PAINT_SIZE / 2.0,
            canvas_y - PAINT_SIZE / 2.0,
            PAINT_SIZE,
            PAINT_SIZE,
        );
    }

    fn notify_change(&self, onchange: &Callback<Vec<u8>>) {
        let context = match &self.draw_context {
            Some(context) => context,
            None => return,
        };

        let img = match context.get_image_data(0.0, 0.0, self.width, self.height) {
            Ok(img) => img,
            Err(err) => {
                error!("DrawableCanvas::notify_change : {err:?}");
                return;
            }
        };

        onchange.emit(img.data().0);
    }

    fn clear(&self) {
        let ctx = match &self.draw_context {
            Some(ctx) => ctx,
            None => {
                warn!("DrawableCanvas::clear ctx isnt initialized cannot clear canvas");
                return;
            }
        };

        ctx.set_fill_style(&JsValue::from(&self.background));
        ctx.fill_rect(0.0, 0.0, self.width, self.height);
    }
}

impl Component for DrawableCanvas {
    type Message = DrawableCanvasMessage;

    type Properties = DrawableCanvasProp;

    fn create(ctx: &Context<Self>) -> Self {
        Self {
            canvas_ref: NodeRef::default(),
            pressed: false,
            draw_context: None,
            canvas: None,
            width: ctx.props().width as f64,
            height: ctx.props().height as f64,
            color: ctx.props().color.clone(),
            background: ctx.props().background.clone(),
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let onmousedown = ctx.link().callback(DrawableCanvasMessage::MouseDown);
        let onmouseup = ctx.link().callback(DrawableCanvasMessage::MouseUp);
        let onmousemove = ctx.link().callback(DrawableCanvasMessage::MouseTick);
        let onclick = ctx.link().callback(|_| DrawableCanvasMessage::ClearCanvas);

        html! {
            <div id={ctx.props().id.clone()}>
                <canvas
                    ref={self.canvas_ref.clone()}
                    id={ctx.props().canvasid.clone()}
                    class={ctx.props().class.clone()}
                    width={format!("{}", ctx.props().width)}
                    height={format!("{}", ctx.props().height)}
                    {onmousedown} {onmouseup} {onmousemove} />
                <div>
                <button class="icon" {onclick}><i class="fas fa-trash"></i></button>
                </div>
            </div>
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            DrawableCanvasMessage::MouseDown(e) => {
                let x = e.offset_x();
                let y = e.offset_y();

                self.draw_on_canvas(x, y);
                self.pressed = true;
            }
            DrawableCanvasMessage::MouseUp(e) => {
                let x = e.offset_x();
                let y = e.offset_y();

                self.draw_on_canvas(x, y);
                self.pressed = false;
                if let Some(onchange) = &ctx.props().onchange {
                    self.notify_change(onchange);
                }
            }
            DrawableCanvasMessage::MouseTick(e) => {
                if !self.pressed {
                    return false;
                }
                let x = e.offset_x();
                let y = e.offset_y();

                self.draw_on_canvas(x, y);
            }
            DrawableCanvasMessage::ClearCanvas => self.clear(),
        }
        false
    }

    fn changed(&mut self, ctx: &Context<Self>) -> bool {
        self.width = ctx.props().width as f64;
        self.height = ctx.props().height as f64;
        true
    }

    fn rendered(&mut self, _ctx: &Context<Self>, _first_render: bool) {
        let canvas = self.canvas_ref.cast::<HtmlCanvasElement>();
        let draw_context = canvas
            .as_ref()
            .and_then(|cv| cv.get_context("2d").ok().and_then(|d| d))
            .and_then(|o| o.dyn_into::<CanvasRenderingContext2d>().ok());

        if canvas.is_none() {
            error!("DrawableCanvas::rendered : Couldnt initialize the canvas");
            return;
        }
        if draw_context.is_none() {
            error!("DrawableCanvas::rendered : Couldnt initialize the context");
            return;
        }

        self.canvas = canvas;
        self.draw_context = draw_context;

        self.clear();
    }
}
