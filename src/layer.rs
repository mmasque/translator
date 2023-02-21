use super::outer;
use crate::activation::Activation;
use drift::float::F64;
// use log::debug;
use ndarray::{Array1, Array2};
use rand::random;

pub struct Layer<F> where 
F: Fn(F64) -> F64 {
    pub weights: Array2<F64>,
    pub biases: Array1<F64>,
    pub activation: F,
}

// #[derive(Debug)]
// pub struct LayerCache<T> {
//     pub weighted_inputs: Array1<T>,
//     pub net_inputs: Array1<T>,
// }

// impl LayerCache<f32> {
//     pub fn new(weighted_inputs: Array1<f32>, net_inputs: Array1<f32>) -> Self {
//         Self {
//             weighted_inputs: weighted_inputs,
//             net_inputs: net_inputs,
//         }
//     }
// }
// want: to note the num_nodes, num_inputs as requirements for the input dimensions of forward
impl <F> Layer<F> where 
F: Fn(F64) -> F64 + Activation{
    pub fn new(activation:F, weights: Array2<f32>, biases: Array1<f32>) -> Self {
        assert_eq!(weights.dim().0, biases.len());
        Self {
            weights: weights,
            biases: biases,
            activation: activation,
        }
    }
    pub fn basic(activation: F, num_nodes: usize, num_inputs: usize) -> Self {
        Self {
            weights: Array2::<F64>::ones((num_nodes, num_inputs)), // todo change to random
            biases: Array1::<F64>::zeros(num_nodes),
            activation: activation,
        }
    }
    //TODO: need to impl random function for F64s
    // pub fn random(activation: F, num_nodes: usize, num_inputs: usize) -> Self {
    //     Self {
    //         weights: Array2::<F64>::from_shape_simple_fn((num_nodes, num_inputs), random::<F64>),
    //         biases: Array1::<F64>::from_shape_simple_fn(num_nodes, random::<F64>),
    //         activation: activation,
    //     }
    // }
    // forward pass
    pub fn forward(&mut self, inputs: &Array1<F64>) -> Array1<F64> {
        let x = self.weights.dot(inputs) + &self.biases;
        let y = self.activation.activate(&x);
        y
    }

    pub fn backpropagation(
        &mut self,
        de_dyj: &Array1<F64>, // change in error as outputs of this layer change
        learning_rate: F64,
    ) -> Array1<F64> {

        // want de/dW. 
        // first we compute the change in the output of the activation as function of (weighted?) input: df/dw.
        // then we use the chain rule to compute the change in error as weighted inputs to the layer change: dC/dw
        // then we can use chain rule again to compute just change in error as raw inputs to this layer change: dC/dx

        // now we have to find the change in error as weights change, we had dw/dx * dC/dw = dC/dx 
        // thus we can compute    

        todo!()
    }
}
