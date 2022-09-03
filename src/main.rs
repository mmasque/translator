mod activation;

use crate::activation::{Activation, Derivative, ReLU};
use layer::Layer;
use ndarray::{Array1, Array2};
mod layer;
mod outer;
fn main() {
    let inputs = Array1::from_vec(vec![-50.0, 2.0, 3.0, 4.0]);
    let inputs2 = Array1::from_vec(vec![-5.0, 5.0, 2.0]);
    let builder = NetworkBuilder::new();
    let mut network = builder.add_layer(3).add_layer(4).add_layer(3).build();
}
// the first goal is to build a simple neural network.

// An input layer
// A hidden layer with ReLU
// an output layer with a sigmoid
// will need to implement forward propagation
// will need to implement backpropagation
// the dataset will be MINST, from here: https://deepai.org/dataset/mnist

struct Network {
    layers: Vec<Layer<f32, ReLU>>,
}

impl Network {
    pub fn new() -> Self {
        Self { layers: vec![] }
    }
    pub fn forward(&self, inputs: &Array1<f32>) {
        // TODO allow inputs to be borrowed in a nicer way, without needing to clone
        let result = self
            .layers
            .iter()
            .fold(inputs.clone(), |prev, x| x.forward(&prev));
    }
}

struct NetworkBuilder {
    network: Network,
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self {
            network: Network::new(),
        }
    }
    pub fn add_layer(mut self, num_nodes: usize) -> Self {
        // get the number of nodes in the previous layer (the last in the list)
        let num_inputs = self
            .network
            .layers
            .last()
            .map(|x| x.biases.len())
            .unwrap_or(num_nodes); // if empty the layer is input nodes

        self.network
            .layers
            .push(Layer::new(ReLU {}, num_nodes, num_inputs));

        self
    }
    pub fn build(self) -> Network {
        self.network
    }
}
