mod activation;
mod layer;
fn main() {

}
// mod network;
// use network::{NetworkBuilder};

// use activation::{
//     Activation, Derivative, Function, Logistic, Loss, ParamDerivative, Softmax, SquareError,
// };
// use mnist::{self, Mnist, MnistBuilder};
// use ndarray::{arr1, arr2, s, Array1, Array2};
mod outer;
// fn main() {
//     let builder = NetworkBuilder::new();
//     // need to be using a softmax at the end.
//     // ooft convoluted Boxes
//     let mut network = builder
//         .add_layer(784, Box::new(Logistic {}))
//         .add_layer(128, Box::new(Logistic {}))
//         .add_layer(64, Box::new(Logistic {}))
//         .add_layer(10, Box::new(Softmax {}))
//         .build();

//     // take in mnist
//     let Mnist {
//         trn_img,
//         trn_lbl,
//         tst_img,
//         tst_lbl,
//         ..
//     } = MnistBuilder::new()
//         .label_format_digit()
//         .training_set_length(50_000)
//         .validation_set_length(10_000)
//         .test_set_length(10_000)
//         .finalize();

//     let train_data = Array2::from_shape_vec((50_000, 784), trn_img)
//         .expect("Error converting images to Array3 struct")
//         .map(|x| *x as f32 / 256.0);

//     // Convert the returned Mnist struct to Array2 format
//     let train_labels: Array1<u8> = Array1::from_shape_vec(50_000, trn_lbl)
//         .expect("Error converting training labels to Array2 struct");

//     for i in 0..5_000 {
//         let image = train_data.slice(s![i, ..]).to_owned();

//         let label_ind = train_labels.get(i).unwrap();
//         let mut label = Array1::<f32>::zeros(10);
//         label[*label_ind as usize] = 1.0;

//         let out = network.forward(&image);
//         let loss = network.backpropagation(&out, &label);
//         println!("| {:?} ", loss);
//     }
// }
