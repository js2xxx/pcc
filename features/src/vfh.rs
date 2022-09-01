use nalgebra::{Scalar, Vector4};

pub struct VfhEstimation<T: Scalar> {
    pub normal: Option<Vector4<T>>,
    pub centroid: Option<Vector4<T>>,
    pub has_size: bool,
}
