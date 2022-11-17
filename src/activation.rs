use ndarray::{Array, Array1};

pub trait Activation {
    fn activate(&self, input: &Array1<f32>) -> Array1<f32>;
}

pub trait Loss {
    fn loss(result: &Array1<f32>, expected: &Array1<f32>) -> f32;
}

pub trait Derivative {
    fn derivative(&self, input: &Array1<f32>) -> Array1<f32>;
}

// I don't like this approach, but we need a derivative function that takes
// in a second array as parameter.
// TODO change so that other traits are also without borrowing self
pub trait ParamDerivative {
    fn derivative(input: &Array1<f32>, param: &Array1<f32>) -> Array1<f32>;
}

#[derive(Debug)]
pub struct ReLU;
pub struct Softmax;
pub struct Logistic;

pub struct SquareError;

impl Activation for ReLU {
    fn activate(&self, input: &Array1<f32>) -> Array1<f32> {
        input.map(|x| if x < &0.0 { 0.0 } else { *x })
    }
}

impl Activation for Softmax {
    fn activate(&self, _input: &Array1<f32>) -> Array1<f32> {
        //let exp_sum: f32 = input.iter().map(|x| x.exp()).sum();
        //input.iter().map(|x| x.exp() / exp_sum).collect()
        unimplemented!()
    }
}

impl Activation for Logistic {
    fn activate(&self, input: &Array1<f32>) -> Array1<f32> {
        input.map(|x| 1.0 / (1.0 + std::f32::consts::E.powf(-*x)))
    }
}

impl Derivative for ReLU {
    fn derivative(&self, input: &Array1<f32>) -> Array1<f32> {
        input.map(|x| if x < &0.0 { 0.0 } else { 1.0 })
    }
}

impl Derivative for Logistic {
    fn derivative(&self, input: &Array1<f32>) -> Array1<f32> {
        input.map(|x| {
            (1.0 / (1.0 + std::f32::consts::E.powf(-*x)))
                * (1.0 - 1.0 / (1.0 + std::f32::consts::E.powf(-*x)))
        })
    }
}

impl Derivative for SquareError {
    fn derivative(&self, input: &Array1<f32>) -> Array1<f32> {
        todo!()
    }
}

impl Loss for SquareError {
    fn loss(result: &Array1<f32>, expected: &Array1<f32>) -> f32 {
        assert_eq!(result.dim(), expected.dim());
        // compute the loss
        result
            .iter()
            .zip(expected.iter())
            .fold(0.0, |total, (x, y)| total + 0.5 * (x - y).powi(2))
    }
}

impl ParamDerivative for SquareError {
    fn derivative(result: &Array1<f32>, expected: &Array1<f32>) -> Array1<f32> {
        result
            .iter()
            .zip(expected.iter())
            .map(|(r, e)| r - e)
            .collect()
    }
}
