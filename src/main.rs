mod activation;
use crate::activation::{Activation, ReLU, Softmax};
use ndarray::{Array1, Array2, ShapeError};
fn main() {
    let test = vec![-1.0, 2.0, 3.0, 4.0];
    let val = ReLU {}.activate(&test);
    println!("{:?}", val);
    println!("{:?}", Softmax {}.activate(&test));
    let t = ReLU {};
}

// the first goal is to build a simple neural network.

// An input layer
// A hidden layer with ReLU
// an output layer with a sigmoid
// will need to implement forward propagation
// will need to implement backpropagation
// the dataset will be MINST, from here: https://deepai.org/dataset/mnist

// this activation layout is already a bit annoying:
pub struct Layer<T, F: Activation, Derivative> {
    pub Weights: Array2<T>,
    pub Biases: Array1<T>,
    pub Activation: F,
}

// what do layers need?
// inputs to each node
// outputs from each node (or none)
// weights
// biases
// activation

// want: to note the num_nodes, num_inputs as requirements for the input dimensions of forward
impl<F: Activation, Derivative> Layer<f32, F> {
    pub fn new(activation: F, num_nodes: usize, num_inputs: usize) -> Self {
        Self {
            Weights: Array2::<f32>::zeros((num_nodes, num_inputs)),
            Biases: Array1::<f32>::zeros(num_nodes),
            Activation: activation,
        }
    }
    // forward pass
    pub fn forward(&self, inputs: Array1<f32>) -> Array1<f32> {
        // could maybe use a Box and std [[]] since then we can provide dimensions and avoid
        // the possible panic from Weights.dot
        self.Weights.dot(&inputs) + &self.Biases
    }

    pub fn backpropagation(&self, inputs: Array1<f32>, dC_dq: Array1<f32>) -> Array1<f32> {
        unimplemented!();
        // we want, for loss C, dC/dw for each weight w, and dC/db for each bias b.
        // chain rule: dq/dw * dC/dq = dc/dw, for q any previous (right to left) ops.
        let i_d = self.Activation.derivative(inputs);
    }
}
