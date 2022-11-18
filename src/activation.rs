use ndarray::{Array, Array1};

pub trait Function: Activation + Derivative {}
impl<T: Activation + Derivative> Function for T {}

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
    fn activate(&self, input: &Array1<f32>) -> Array1<f32> {
        // TODO avoid this double computation by looking into a non consuming sum.
        let exps = input.iter().map(|x| x.exp());
        let sum: f32 = input.iter().map(|x| x.exp()).sum();
        exps.map(|x| x / sum).collect()
    }
}

impl Activation for Logistic {
    fn activate(&self, input: &Array1<f32>) -> Array1<f32> {
        input.map(|x| 1.0 / (1.0 + (-*x).exp()))
    }
}

impl Derivative for ReLU {
    fn derivative(&self, input: &Array1<f32>) -> Array1<f32> {
        input.map(|x| if x < &0.0 { 0.0 } else { 1.0 })
    }
}

impl Derivative for Logistic {
    fn derivative(&self, input: &Array1<f32>) -> Array1<f32> {
        input.map(|x| (1.0 / (1.0 + (-*x).exp())) * (1.0 - 1.0 / (1.0 + (-*x).exp())))
    }
}

impl Derivative for Softmax {
    // Softmax is not expressible as an elementwise function
    // instead it's a proper vector function so its derivative
    // is the jacobian. So I'll just ignore the off diagonal
    // elements https://deepnotes.io/softmax-crossentropy#derivative-of-softmax
    fn derivative(&self, input: &Array1<f32>) -> Array1<f32> {
        let sum_squared: f32 = f32::powi(input.iter().map(|x| x.exp()).sum(), 2);
        input.map(|x| f32::powi(*x, 2) / sum_squared)
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
