use nalgebra::{RealField, Rotation3, Scalar, Vector4};
use pcc_common::{
    filter::{ApproxFilter, Filter},
    point::Point,
    point_cloud::PointCloud,
};
use pcc_sac::Plane;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CropBox<T: RealField> {
    pub min: Vector4<T>,
    pub max: Vector4<T>,
    pub rotation: Rotation3<T>,
    pub negative: bool,
}

impl<T: RealField> CropBox<T> {
    pub fn new(
        min: Vector4<T>,
        max: Vector4<T>,
        rotation: Rotation3<T>,
        negative: bool,
    ) -> CropBox<T> {
        CropBox {
            min,
            max,
            rotation,
            negative,
        }
    }
}

impl<T: RealField, P: Point<Data = T>> Filter<[P]> for CropBox<T> {
    fn filter_indices(&mut self, input: &[P]) -> Vec<usize> {
        let center = (&self.min + &self.max).unscale(T::one() + T::one()).xyz();

        let mut indices = (0..input.len()).collect::<Vec<_>>();
        indices.retain(|&index| {
            let coords = &input[index].coords().xyz();
            let delta = coords - &center;
            let local_delta = self.rotation.inverse_transform_vector(&delta);
            let local_coords = local_delta + &center;

            (self.min.xyz() <= local_coords && local_coords <= self.max.xyz()) ^ self.negative
        });
        indices
    }

    fn filter_all_indices(&mut self, input: &[P]) -> (Vec<usize>, Vec<usize>) {
        let center = (&self.min + &self.max).unscale(T::one() + T::one()).xyz();

        let mut indices = (0..input.len()).collect::<Vec<_>>();
        let mut removed = Vec::with_capacity(indices.len());
        indices.retain(|&index| {
            let coords = &input[index].coords().xyz();
            let delta = coords - &center;
            let local_delta = self.rotation.inverse_transform_vector(&delta);
            let local_coords = local_delta + &center;

            let ret =
                (self.min.xyz() <= local_coords && local_coords <= self.max.xyz()) ^ self.negative;
            if ret {
                removed.push(index);
            }
            ret
        });
        (indices, removed)
    }
}

impl<T: RealField, P: Point<Data = T>> ApproxFilter<PointCloud<P>> for CropBox<T> {
    fn filter(&mut self, input: &PointCloud<P>) -> PointCloud<P> {
        let center = (&self.min + &self.max).unscale(T::one() + T::one()).xyz();

        let mut storage = Vec::from(&**input);
        storage.retain(|point| {
            let coords = &point.coords().xyz();
            let delta = coords - &center;
            let local_delta = self.rotation.inverse_transform_vector(&delta);
            let local_coords = local_delta + &center;

            (self.min.xyz() <= local_coords && local_coords <= self.max.xyz()) ^ self.negative
        });
        PointCloud::from_vec(storage, 1)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CropPlane<T: Scalar> {
    pub plane: Plane<T>,
}

impl<T: RealField> CropPlane<T> {
    fn inner<P: Point<Data = T>>(&self) -> impl FnMut(&P) -> bool + '_ {
        |point| self.plane.same_side_with_normal(point.coords())
    }
}

impl<T: RealField, P: Point<Data = T>> Filter<[P]> for CropPlane<T> {
    fn filter_indices(&mut self, input: &[P]) -> Vec<usize> {
        self.inner().filter_indices(input)
    }

    fn filter_all_indices(&mut self, input: &[P]) -> (Vec<usize>, Vec<usize>) {
        self.inner().filter_all_indices(input)
    }
}

impl<T: RealField, P: Point<Data = T>> ApproxFilter<PointCloud<P>> for CropPlane<T> {
    fn filter(&mut self, input: &PointCloud<P>) -> PointCloud<P> {
        self.inner().filter(input)
    }
}
