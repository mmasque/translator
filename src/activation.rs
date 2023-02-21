use ndarray::{Array, Array1};
use drift::float::F64;
use num_traits::float::Float;
pub trait Activation {
    fn activate(&self, input: &Array1<F64>) -> Array1<F64>;
}

pub trait Loss {
    fn loss(result: &Array1<F64>, expected: &Array1<F64>) -> F64;
}

#[derive(Debug)]
pub struct ReLU;
pub struct Softmax;
pub struct Logistic;

pub struct SquareError;

impl Activation for ReLU {
    fn activate(&self, input: &Array1<F64>) -> Array1<F64> {
        input.map(|x| if x < &F64::c(0.0) { F64::c(0.0) } else { *x })
    }
}

impl Activation for Softmax {
    fn activate(&self, input: &Array1<F64>) -> Array1<F64> {
        let exps: Array1<F64> = input.iter().map(|x| x.exp()).collect();
        let sum = exps.iter().fold(F64::c(0.0), |i, x| i + x.exp());
        exps.iter().map(|x| x / sum).collect()
    }
}

impl Activation for Logistic {
    fn activate(&self, input: &Array1<F64>) -> Array1<F64> {
        input.map(|x| F64::c(1.0) / (F64::c(1.0) + (-*x).exp()))
    }
}

impl Loss for SquareError {
    fn loss(result: &Array1<F64>, expected: &Array1<F64>) -> F64 {
        assert_eq!(result.dim(), expected.dim());
        // compute the loss
        result
            .iter()
            .zip(expected.iter())
            .fold(F64::c(0.0), |total, (x, y)| total + F64::c(0.5) * (x - y).powi(2))
    }
}

