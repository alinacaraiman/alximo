use anyhow::{ensure, Result};
use maths::squared_sum;
use ndarray::Array1;
use statrs::distribution::{Beta, ContinuousCDF};

mod maths;

pub fn pearsonr(
    x: Array1<f64>,
    y: Array1<f64>,
    alternative: Option<&str>,
) -> Result<(f64, f64)> {
    ensure!(x.len() == x.len(), "x and y have different lenghts. To calculate the Pearson R coefficient, x and y should have the same length");
    let n = x.len() as f64;

    let (sum_x, sum_y) = (x.sum(), y.sum());

    let (squared_x_sum, squared_y_sum) = (squared_sum(&x), squared_sum(&y));

    let cross_product_sum = (x * y).sum();

    let numinator = n * cross_product_sum - (sum_x * sum_y);
    let denominator =
        ((n * squared_x_sum - sum_x.powi(2)) * (n * squared_y_sum - sum_y.powi(2))).sqrt();

    let mut r = numinator / denominator;

    r = r.min(1.).max(-1.);

    let alternative = alternative.unwrap_or("two-sided");

    let p_val = pearson_significance(r, n, alternative);

    Ok((r, p_val))
}

fn pearson_significance(r: f64, n: f64, alternative: &str) -> f64 {
    let ab = (n / 2.) - 1.;
    let beta = Beta::new(ab, ab).unwrap();

    match alternative {
        "two-sided" => beta.sf(r.abs()) * 2.,
        "less" => beta.cdf(r),
        "greater" => beta.sf(r),
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array;
    use statrs::assert_almost_eq;

    use crate::pearsonr;

    #[test]
    fn test_pearson_corr() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = Array::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

        let (r, p) = pearsonr(x, y, None).unwrap();
        println!("Pearson correlation coefficient: {}, Probability {}", r, p);
        assert_almost_eq!(r, 1., 1e-14);
        assert_almost_eq!(p, 0., 1e-14);
    }
}
