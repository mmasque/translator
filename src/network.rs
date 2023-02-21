use ndarray::{Array1, Array2};

use crate::{layer::Layer, activation::{SquareError, Function, Loss, ParamDerivative}};


pub struct Network {
    layers: Vec<Layer<f32>>,
}

impl Network {
    pub fn new() -> Self {
        Self { layers: vec![] }
    }
    pub fn forward(&mut self, inputs: &Array1<f32>) -> Array1<f32> {
        // TODO allow inputs to be borrowed in a nicer way, without needing to clone
        self.layers
            .iter_mut()
            .fold(inputs.clone(), |prev, x| x.forward(&prev))
    }
    pub fn backpropagation(&mut self, result: &Array1<f32>, expected: &Array1<f32>) -> f32 {
        // backprop as used here is a misnomer, because we are combining it with gradient descent to update weights and biases.
        //TODO extract to a Loss enum to support multiple enum types
        let loss = SquareError::loss(result, expected);

        // compute change in loss wrt output layer
        let de_dr = SquareError::derivative(result, expected);

        self.layers
            .iter_mut()
            .rev()
            .fold(de_dr, |de_dyj, layer| layer.backpropagation(&de_dyj, 0.5));
        loss
    }
}

pub struct NetworkBuilder {
    network: Network,
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self {
            network: Network::new(),
        }
    }
    pub fn add_layer(mut self, num_nodes: usize, act: Box<dyn Function>) -> Self {
        // get the number of nodes in the previous layer (the last in the list)
        let num_inputs = self
            .network
            .layers
            .last()
            .map(|x| x.biases.len())
            .unwrap_or(num_nodes); // if empty the layer is input nodes

        self.network
            .layers
            .push(Layer::random(act, num_nodes, num_inputs));

        self
    }
    pub fn add_layer_manually(
        mut self,
        weights: Array2<f32>,
        biases: Array1<f32>,
        act: Box<dyn Function>,
    ) -> Self {
        self.network.layers.push(Layer::new(act, weights, biases));
        self
    }
    pub fn build(self) -> Network {
        self.network
    }
}