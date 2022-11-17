use super::outer;
use crate::activation::{Activation, Derivative};
use ndarray::{Array1, Array2};

#[derive(Debug)]
pub struct Layer<T, F: Activation + Derivative> {
    pub weights: Array2<T>,
    pub biases: Array1<T>,
    pub activation: F,
    pub cache: Option<LayerCache<T>>,
}

#[derive(Debug)]
pub struct LayerCache<T> {
    pub weigthed_inputs: Array1<T>,
    pub net_inputs: Array1<T>,
}

impl LayerCache<f32> {
    pub fn new(weigthed_inputs: Array1<f32>, net_inputs: Array1<f32>) -> Self {
        Self {
            weigthed_inputs: weigthed_inputs,
            net_inputs: net_inputs,
        }
    }
}
// want: to note the num_nodes, num_inputs as requirements for the input dimensions of forward
impl<F: Activation + Derivative> Layer<f32, F> {
    pub fn new(activation: F, weights: Array2<f32>, biases: Array1<f32>) -> Self {
        assert_eq!(weights.dim().0, biases.len());
        Self {
            weights: weights,
            biases: biases,
            activation: activation,
            cache: None,
        }
    }
    pub fn basic(activation: F, num_nodes: usize, num_inputs: usize) -> Self {
        Self {
            weights: Array2::<f32>::ones((num_nodes, num_inputs)), // todo change to random
            biases: Array1::<f32>::zeros(num_nodes),
            activation: activation,
            cache: None,
        }
    }
    // forward pass
    pub fn forward(&mut self, inputs: &Array1<f32>) -> Array1<f32> {
        let x = self.weights.dot(inputs) + &self.biases;
        let y = self.activation.activate(&x);
        self.cache = Some(LayerCache::new(x, inputs.clone()));
        y // TODO try to avoid the clone ineficiency
    }

    pub fn backpropagation(
        &mut self,
        de_dyj: &Array1<f32>, // change in error as outputs of this layer change
        // x: &Array1<f32>, // weighed sum of inputs to this layer (this is the input to activation f)
        // y: &Array1<f32>, // outputs of previous layer
        learning_rate: f32,
    ) -> Array1<f32> {
        // y is output O, x is net (x is activation inv of y).
        // dE_dyj is the change in Error (cost) with respect to the outputs of the previous layer j, yj.
        // We'll compute dE_dyi (i the current layer), and along the way find dE/dw for the weights in this layer.
        let cache = self.cache.as_ref().unwrap();
        println!(
            "cache weighted_inputs: {:?}, cache outputs: {:?}",
            cache.weigthed_inputs, cache.net_inputs
        );
        let dyj_dxj = self.activation.derivative(&cache.weigthed_inputs);
        println!("The change of y wr x is: {:?}", dyj_dxj);

        let de_dxj = de_dyj * dyj_dxj;
        println!("The change of E wr xj is: {:?}", de_dxj);
        // this is probably slow. computes weight diffs
        let de_dwji = outer::outer(&de_dxj, &cache.net_inputs);

        println!("The change of E wr wji is: {:?}", de_dwji);
        // compute dE_dyi
        let de_dyi = self.weights.dot(&de_dxj);

        println!("The change of E wr yi is: {:?}", de_dyi);
        // update the weights
        self.weights = &self.weights - learning_rate * de_dwji;
        println!("The weights are now: {:?}", self.weights);
        // bias updates
        self.biases = &self.biases - learning_rate * de_dxj;
        println!("The biases are now: {:?}", self.biases);

        de_dyi
    }
}
