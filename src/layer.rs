use super::outer;
use crate::activation::{Activation, Derivative, ReLU};
use ndarray::{Array1, Array2};
pub struct Layer<T, F: Activation + Derivative> {
    pub weights: Array2<T>,
    pub biases: Array1<T>,
    pub activation: F,
}

// want: to note the num_nodes, num_inputs as requirements for the input dimensions of forward
impl<F: Activation + Derivative> Layer<f32, F> {
    pub fn new(activation: F, num_nodes: usize, num_inputs: usize) -> Self {
        Self {
            weights: Array2::<f32>::ones((num_nodes, num_inputs)), // todo change to random
            biases: Array1::<f32>::zeros(num_nodes),
            activation: activation,
        }
    }
    // forward pass
    pub fn forward(&self, inputs: &Array1<f32>) -> Array1<f32> {
        // could maybe use a Box and std [[]] since then we can provide dimensions and avoid
        // the possible panic from Weights.dot
        let x = self.weights.dot(inputs) + &self.biases;
        self.activation.activate(&x)
    }

    pub fn backpropagation(
        &mut self,
        de_dyj: &Array1<f32>, // change in error as outputs of this layer change
        x: &Array1<f32>, // weighed sum of inputs to this layer (this is the input to activation f)
        y: &Array1<f32>, // outputs of previous layer
        learning_rate: f32,
    ) -> Array1<f32> {
        // dE_dyj is the change in Error (cost) with respect to the outputs of the previous layer j, yj.
        // We'll compute dE_dyi (i the current layer), and along the way find dE/dw for the weights in this layer.
        let dyj_dxj = self.activation.derivative(&x);
        println!("The change of y wr x is: {:?}", dyj_dxj);

        let de_dxj = de_dyj * dyj_dxj;
        println!("The change of E wr xj is: {:?}", de_dxj);
        // this is probably slow. computes weight diffs
        let de_dwji = outer::outer(&de_dxj, y);

        println!("The change of E wr wji is: {:?}", de_dwji);
        // compute dE_dyi
        let de_dyi = self.weights.dot(&de_dxj);

        println!("The change of E wr yi is: {:?}", de_dyi);
        // update the weights
        self.weights = &self.weights - learning_rate * de_dwji;

        // bias updates
        self.biases = &self.biases - learning_rate * de_dxj;

        de_dyi
    }
}
