mod activation;

use activation::{Logistic, Loss, ParamDerivative, SquareError};
use layer::Layer;
use ndarray::{arr1, arr2, Array1, Array2};
mod layer;
mod outer;
fn main() {
    let inputs = Array1::from_vec(vec![0.05, 0.1]);
    let expected = Array1::from_vec(vec![0.01, 0.99]);
    let builder = NetworkBuilder::new();
    let mut network = builder
        .add_layer_manually(arr2(&[[0.15, 0.2], [0.25, 0.3]]), arr1(&[0.35, 0.35]))
        .add_layer_manually(arr2(&[[0.4, 0.45], [0.5, 0.55]]), arr1(&[0.6, 0.6]))
        .build();
    let out = network.forward(&inputs);
    let loss = network.backpropagation(&out, &expected);
    println!("out {:?}; loss {:?}", &out, &loss);
}
// the first goal is to build a simple neural network.

// An input layer
// A hidden layer with ReLU
// an output layer with a sigmoid
// will need to implement forward propagation
// will need to implement backpropagation
// the dataset will be MINST, from here: https://deepai.org/dataset/mnist

struct Network {
    layers: Vec<Layer<f32, Logistic>>,
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
    pub fn backpropagation(&mut self, result: &Array1<f32>, expected: &Array1<f32>) {
        // backprop as used here is a misnomer, because we are combining it with gradient descent to update weights and biases.
        //TODO extract to a Loss enum to support multiple enum types
        let loss = SquareError::loss(result, expected);
        println!("Loss: {:?}", loss);
        // compute change in loss wrt output layer
        let de_dr = SquareError::derivative(result, expected);
        println!("{:?}", de_dr);

        //
        self.layers
            .iter_mut()
            .rev()
            .fold(de_dr, |de_dyj, layer| layer.backpropagation(&de_dyj, 0.5));
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
            .push(Layer::basic(Logistic {}, num_nodes, num_inputs));

        self
    }
    pub fn add_layer_manually(mut self, weights: Array2<f32>, biases: Array1<f32>) -> Self {
        self.network
            .layers
            .push(Layer::new(Logistic {}, weights, biases));
        self
    }
    pub fn build(self) -> Network {
        self.network
    }
}
