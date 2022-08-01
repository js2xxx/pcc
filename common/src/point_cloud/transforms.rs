use nalgebra::{ClosedAdd, ClosedMul, Matrix4, RealField, Scalar, TCategory, Vector4};
use num::Num;

pub trait Transform<T: Scalar> {
    fn so3(&self, from: &Vector4<T>, to: &mut Vector4<T>);

    fn se3(&self, from: &Vector4<T>, to: &mut Vector4<T>);
}

impl<T: Scalar + Num + Copy + ClosedAdd + ClosedMul> Transform<T> for Matrix4<T> {
    fn so3(&self, from: &Vector4<T>, to: &mut Vector4<T>) {
        *to = self * Vector4::from([from[0], from[1], from[2], T::zero()]);
    }

    fn se3(&self, from: &Vector4<T>, to: &mut Vector4<T>) {
        *to = self * Vector4::from([from[0], from[1], from[2], T::one()]);
    }
}

impl<T: Scalar + Num + Copy + ClosedAdd + ClosedMul + RealField, C: TCategory> Transform<T>
    for nalgebra::Transform<T, C, 3>
{
    fn so3(&self, from: &Vector4<T>, to: &mut Vector4<T>) {
        self.matrix().so3(from, to)
    }

    fn se3(&self, from: &Vector4<T>, to: &mut Vector4<T>) {
        self.matrix().se3(from, to)
    }
}
