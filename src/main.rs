mod activation;
use activation::{
    Activation, Derivative, Function, Logistic, Loss, ParamDerivative, Softmax, SquareError,
};
use layer::Layer;
use mnist::{self, Mnist, MnistBuilder};
use ndarray::{arr1, arr2, s, Array1, Array2};
mod layer;
mod outer;
fn main() {
    let builder = NetworkBuilder::new();
    // need to be using a softmax at the end.
    // ooft convoluted Boxes
    let mut network = builder
        .add_layer(784, Box::new(Logistic {}))
        .add_layer(128, Box::new(Logistic {}))
        .add_layer(64, Box::new(Logistic {}))
        .add_layer(10, Box::new(Softmax {}))
        .build();

    // take in mnist
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let train_data = Array2::from_shape_vec((50_000, 784), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array1<u8> = Array1::from_shape_vec(50_000, trn_lbl)
        .expect("Error converting training labels to Array2 struct");

    for i in 0..5_000 {
        let image = train_data.slice(s![i, ..]).to_owned();

        let label_ind = train_labels.get(i).unwrap();
        let mut label = Array1::<f32>::zeros(10);
        label[*label_ind as usize] = 1.0;

        let out = network.forward(&image);
        let loss = network.backpropagation(&out, &label);
        println!("| {:?} ", loss);
    }
}

struct Network {
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

struct NetworkBuilder {
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
