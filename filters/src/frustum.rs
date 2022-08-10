use std::{array, fmt::Debug};

use nalgebra::{matrix, RealField, Transform3};
use pcc_common::{
    filter::{ApproxFilter, Filter},
    point_cloud::PointCloud,
    points::Point3Infoed,
};
use pcc_sac::{Plane, PlaneEstimator};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FrustumCulling<T: RealField> {
    /// The value must be greater than zero and less than PI / 2 in radians.
    pub vertical_fov: T,
    /// The value must be greater than zero and less than PI / 2 in radians.
    pub horizontal_fov: T,
    pub near_distance: T,
    pub far_distance: T,
    /// The value must be between -1 and 1.
    pub vertical_roi_min: T,
    /// The value must be between -1 and 1.
    pub vertical_roi_max: T,
    /// The value must be between -1 and 1.
    pub horizontal_roi_min: T,
    /// The value must be between -1 and 1.
    pub horizontal_roi_max: T,
    pub camera_pose: Transform3<T>,
}

impl<T: RealField> FrustumCulling<T> {
    /// NOTE: All the normal vectors of returned planes point to the inside of
    /// the frustum.
    pub fn compute_planes(&self) -> [Plane<T>; 6] {
        assert!(T::zero() < self.horizontal_fov && self.horizontal_fov < T::frac_pi_2());
        assert!(T::zero() < self.vertical_fov && self.vertical_fov < T::frac_pi_2());
        assert!(T::zero() < self.near_distance && self.near_distance < self.far_distance);
        assert!(
            -T::one() <= self.horizontal_roi_min
                && self.horizontal_roi_min < self.horizontal_roi_max
                && self.horizontal_roi_max <= T::one()
        );
        assert!(
            -T::one() <= self.vertical_roi_min
                && self.vertical_roi_min < self.vertical_roi_max
                && self.vertical_roi_min <= T::one()
        );

        let near_center = matrix![self.near_distance.clone(); T::zero(); T::zero()];
        let far_center = matrix![self.far_distance.clone(); T::zero(); T::zero()];

        let width_mul = self.horizontal_fov.clone().tan();
        let height_mul = self.vertical_fov.clone().tan();

        let near_y_max =
            self.near_distance.clone() * width_mul.clone() * self.horizontal_roi_max.clone();
        let near_y_min =
            self.near_distance.clone() * width_mul.clone() * self.horizontal_roi_min.clone();
        let near_z_max =
            self.near_distance.clone() * height_mul.clone() * self.vertical_roi_max.clone();
        let near_z_min =
            self.near_distance.clone() * height_mul.clone() * self.vertical_roi_min.clone();

        let far_y_max =
            self.far_distance.clone() * width_mul.clone() * self.horizontal_roi_max.clone();
        let far_y_min = self.far_distance.clone() * width_mul * self.horizontal_roi_min.clone();
        let far_z_max =
            self.far_distance.clone() * height_mul.clone() * self.vertical_roi_max.clone();
        let far_z_min = self.far_distance.clone() * height_mul * self.vertical_roi_min.clone();

        let points = [
            &near_center + matrix![T::zero(); near_y_max.clone(); near_z_min.clone()], /* Near, left, bottom */
            &near_center + matrix![T::zero(); near_y_min.clone(); near_z_min], /* Near, right,
                                                                                * bottom */
            &near_center + matrix![T::zero(); near_y_max; near_z_max.clone()], // Near, left, top
            near_center + matrix![T::zero(); near_y_min; near_z_max],          // Near, right, top
            &far_center + matrix![T::zero(); far_y_max.clone(); far_z_min.clone()], /* Far, left,
                                                                                * bottom */
            &far_center + matrix![T::zero(); far_y_min; far_z_min], // Far, right, bottom
            &far_center + matrix![T::zero(); far_y_max; far_z_max], // Far, left, top
        ];

        let points: [_; 7] = array::from_fn(|index| {
            self.camera_pose
                .transform_vector(&points[index])
                .insert_row(3, T::one())
        });

        [
            PlaneEstimator::make(&points[0], &points[1], &points[2]), // Near
            PlaneEstimator::make(&points[5], &points[4], &points[6]), // Far
            PlaneEstimator::make(&points[2], &points[0], &points[4]), // Left
            PlaneEstimator::make(&points[3], &points[5], &points[1]), // Right
            PlaneEstimator::make(&points[0], &points[1], &points[5]), // Bottom
            PlaneEstimator::make(&points[2], &points[6], &points[3]), // Top
        ]
    }
}

impl<T: RealField, I> Filter<[Point3Infoed<T, I>]> for FrustumCulling<T> {
    fn filter_indices(&mut self, input: &[Point3Infoed<T, I>]) -> Vec<usize> {
        let planes = self.compute_planes();

        let mut indices = (0..input.len()).collect::<Vec<_>>();
        indices.retain(|&index| {
            { planes.iter() }.all(|plane| plane.same_side_with_normal(&input[index].coords))
        });
        indices
    }

    fn filter_all_indices(&mut self, input: &[Point3Infoed<T, I>]) -> (Vec<usize>, Vec<usize>) {
        let planes = self.compute_planes();

        let mut indices = (0..input.len()).collect::<Vec<_>>();
        let mut removed = Vec::with_capacity(indices.len());
        indices.retain(|&index| {
            let ret =
                { planes.iter() }.all(|plane| plane.same_side_with_normal(&input[index].coords));
            if !ret {
                removed.push(index)
            }
            ret
        });
        (indices, removed)
    }
}

impl<T: RealField, I: Clone + Debug> ApproxFilter<PointCloud<Point3Infoed<T, I>>>
    for FrustumCulling<T>
{
    fn filter(&mut self, input: &PointCloud<Point3Infoed<T, I>>) -> PointCloud<Point3Infoed<T, I>> {
        let planes = self.compute_planes();

        let mut storage = Vec::from(&**input);
        storage.retain(|point| {
            { planes.iter() }.all(|plane| plane.same_side_with_normal(&point.coords))
        });

        PointCloud::from_vec(storage, 1)
    }
}
