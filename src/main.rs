mod activation;
use activation::{Logistic, Loss, ParamDerivative, SquareError};
use layer::Layer;
use mnist::{self, Mnist, MnistBuilder};
use ndarray::{arr1, arr2, s, Array1, Array2};
mod layer;
mod outer;
fn main() {
    // to verify correctness I'm using https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    // which walks through the weight values at every step etc
    // let inputs = Array1::from_vec(vec![0.05, 0.1]);
    // let expected = Array1::from_vec(vec![0.01, 0.99]);
    // let builder = NetworkBuilder::new();
    // let mut network = builder
    //     .add_layer_manually(arr2(&[[0.15, 0.2], [0.25, 0.3]]), arr1(&[0.35, 0.35]))
    //     .add_layer_manually(arr2(&[[0.4, 0.45], [0.5, 0.55]]), arr1(&[0.6, 0.6]))
    //     .build();
    // // just do the same thing a bunch and see if loss goes down
    // for i in (0..1000) {
    //     let out = network.forward(&inputs);
    //     let loss = network.backpropagation(&out, &expected);
    //     print!("| {:?} ", loss);
    // }
    let builder = NetworkBuilder::new();
    let mut network = builder
        .add_layer(784)
        .add_layer(128)
        .add_layer(64)
        .add_layer(10)
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

    for i in 0..50_000 {
        let image = train_data.slice(s![i, ..]).to_owned();
        println!("Image dimension: {:?}", image.dim());
        let label_ind = train_labels.get(i).unwrap();
        let mut label = Array1::<f32>::zeros(10);
        label[*label_ind as usize] = 1.0;
        println!("Label: {:?}", label);
        let out = network.forward(&image);
        println!("Output: {:?}", out);
        let loss = network.backpropagation(&out, &label);
        print!("| {:?} ", loss);
    }
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
    pub fn backpropagation(&mut self, result: &Array1<f32>, expected: &Array1<f32>) -> f32 {
        // backprop as used here is a misnomer, because we are combining it with gradient descent to update weights and biases.
        //TODO extract to a Loss enum to support multiple enum types
        let loss = SquareError::loss(result, expected);
        // compute change in loss wrt output layer
        let de_dr = SquareError::derivative(result, expected);

        //
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
            .push(Layer::random(Logistic {}, num_nodes, num_inputs));

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
