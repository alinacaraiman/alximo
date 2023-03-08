use std::ops::Div;

use alximo_core::utils::{squeeze, squeeze_both};
use anyhow::Result;
use ndarray::{concatenate, Array, Array0, Array1, Array2, Axis};

pub fn squared_sum(x: &Array1<f64>) -> f64 {
    let squared_x = Array::from_vec(x.iter().map(|x| x * x).collect::<Vec<f64>>());
    squared_x.sum()
}

pub fn cov(
    m: Array2<f64>,
    y: Option<Array2<f64>>,
    rowvar: bool,
    bias: bool,
    ddof: Option<usize>,
) -> Result<Array2<f64>> {
    let array = if !rowvar && m.shape()[0] != 1 {
        m.t().to_owned()
    } else {
        m
    };
    let array = if let Some(y) = y {
        let y = if !rowvar && y.shape()[0] != 1 {
            y.t().to_owned()
        } else {
            y
        };
        concatenate![Axis(0), array, y]
    } else {
        array
    };

    let ddof = if let Some(ddof) = ddof {
        ddof
    } else if bias {
        0
    } else {
        1
    };

    let avg_array = Array1::from(vec![array.mean().unwrap()]);
    let fact = array.shape()[1] - ddof;
    let avg_array = avg_array.insert_axis(Axis(1));
    let array = array - avg_array;
    let array_t = array.t().to_owned();
    let mult = 1. / fact as f64;
    let c = array.dot(&array_t) * mult;
    Ok(c)
}

pub fn corrcoef(c: Array2<f64>) -> Result<Array2<f64>> {
    match squeeze(c.clone()) {
        Some(c) => Ok(continue_corr_coef_1d(c)),
        None => match squeeze_both(c.clone()) {
            Some(c) => Ok(continue_corr_coef_0d(c)),
            None => Ok(continue_corr_coef_2d(c)),
        },
    }
}

fn continue_corr_coef_2d(c: Array2<f64>) -> Array2<f64> {
    let d = c.diag();
    let stddev = d.mapv(f64::sqrt);
    let c = c.div(stddev.clone().insert_axis(Axis(1)));
    let c = c / stddev.insert_axis(Axis(0));
    c.mapv(|x| x.min(1.).max(-1.))
}

fn continue_corr_coef_1d(c: Array1<f64>) -> Array2<f64> {
    let d = c.diag();
    let stddev = d.mapv(f64::sqrt);
    let c = c / stddev.clone().insert_axis(Axis(1));
    let c = c / stddev.insert_axis(Axis(0));
    c.mapv(|x| x.min(1.).max(-1.))
}

fn continue_corr_coef_0d(c: Array0<f64>) -> Array2<f64> {
    let d = c.diag();
    let stddev = d.mapv(f64::sqrt);
    let c = c / stddev.clone().insert_axis(Axis(1));
    let c = c / stddev.insert_axis(Axis(0));
    c.mapv(|x| x.min(1.).max(-1.))
}
