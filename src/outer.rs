use ndarray::{Array1, Array2};

pub fn outer(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    // this is not performant, come back TODO
    let mut r = Array2::<f32>::zeros((a.len(), b.len()));
    for i in 0..a.len() {
        for j in 0..b.len() {
            r[[i, j]] = a[i] * b[j];
        }
    }
    r
}
