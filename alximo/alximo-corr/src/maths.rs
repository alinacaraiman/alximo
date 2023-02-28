use ndarray::{Array, Array1};

pub fn squared_sum(x: &Array1<f64>) -> f64 {
    let squared_x = Array::from_vec(x.iter().map(|x| x * x).collect::<Vec<f64>>());
    squared_x.sum()
}
