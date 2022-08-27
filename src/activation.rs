use ndarray::{Array1, Array2, ShapeError};

pub trait Activation {
    fn activate(&self, input: &Array1<f32>) -> Array1<f32>;
}

pub trait Derivative {
    fn derivative(&self, input: &Array1<f32>) -> Array1<f32>;
}

pub struct ReLU;
pub struct Softmax;

impl Activation for ReLU {
    fn activate(&self, input: &Array1<f32>) -> Array1<f32> {
        input.map(|x| if x < &0.0 { 0.0 } else { *x })
    }
}

impl Activation for Softmax {
    fn activate(&self, input: &Array1<f32>) -> Array1<f32> {
        //let exp_sum: f32 = input.iter().map(|x| x.exp()).sum();
        //input.iter().map(|x| x.exp() / exp_sum).collect()
        unimplemented!()
    }
}

impl Derivative for ReLU {
    fn derivative(&self, input: &Array1<f32>) -> Array1<f32> {
        input.map(|x| if x < &0.0 { 0.0 } else { 1.0 })
    }
}
