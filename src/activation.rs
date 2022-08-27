pub trait Activation {
    fn activate(input: &Vec<f32>) -> Vec<f32>;
}

pub struct ReLU;
pub struct Softmax;

impl Activation for ReLU {
    fn activate(input: &Vec<f32>) -> Vec<f32> {
        vec![input.iter().map(|x| if x < &0.0 { 0.0 } else { *x }).sum()]
    }
}
impl Activation for Softmax {
    fn activate(input: &Vec<f32>) -> Vec<f32> {
        let exp_sum: f32 = input.iter().map(|x| x.exp()).sum();
        input.iter().map(|x| x.exp() / exp_sum).collect()
    }
}
