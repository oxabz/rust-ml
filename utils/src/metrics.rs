use tch::Tensor;
use tch_macros_utils::dims;

#[dims(y_hat(B,W,H), y(B,W,H))]
pub fn dice_score_1c(y_hat: &Tensor, y:&Tensor)->Tensor{
    let y_hat = y_hat.to_kind(tch::Kind::Float).sigmoid();
    let y = y.to_kind(tch::Kind::Float);
    let intersect = y_hat.multiply(&y).sum_dim_intlist(&[1, 2], false, tch::Kind::Float);
    let union = y_hat.sum_dim_intlist(&[1, 2], false, tch::Kind::Float) + y.sum_dim_intlist(&[1, 2], false, tch::Kind::Float);
    2.0 as f32 *intersect + union
}