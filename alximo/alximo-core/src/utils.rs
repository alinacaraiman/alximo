use ndarray::{Array0, Array1, Array2, Axis};

pub fn squeeze<A>(x: Array2<A>) -> Option<Array1<A>>
where
    A: Clone,
{
    let shape = x.shape();
    if shape.contains(&1) {
        if shape[0] == 1 {
            Some(x.remove_axis(Axis(0)))
        } else {
            Some(x.remove_axis(Axis(1)))
        }
    } else {
        None
    }
}

pub fn squeeze_both<A>(x: Array2<A>) -> Option<Array0<A>>
where
    A: Clone,
{
    let shape = x.shape();
    if shape[0] == 1 && shape[1] == 1 {
        Some(x.remove_axis(Axis(0)).remove_axis(Axis(0)))
    } else {
        None
    }
}
