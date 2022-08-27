mod activation;
use crate::activation::{Activation, ReLU, Softmax};
fn main() {
    let test = vec![-1.0, 2.0, 3.0, 4.0];
    let val = ReLU::activate(&test);
    println!("{:?}", val);
    println!("{:?}", Softmax::activate(&test));
}

// the first goal is to build a simple neural network.

// An input layer
// A hidden layer with ReLU
// an output layer with a sigmoid
// will need to implement forward propagation
// will need to implement backpropagation
// the dataset will be MINST, from here: https://deepai.org/dataset/mnist
