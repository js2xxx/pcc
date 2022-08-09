use nalgebra::ComplexField;
use pcc_common::{
    filter::{ApproxFilter, Filter},
    point_cloud::PointCloud,
    points::Point3Infoed,
};

pub struct Simple<'a, T> {
    pub constraint: Box<dyn FnMut(&T) -> bool + 'a>,
}

impl<'a, T> Simple<'a, T> {
    pub fn new(constraint: impl Fn(&T) -> bool + 'a) -> Self {
        Simple {
            constraint: Box::new(constraint),
        }
    }
}

impl<'a, T: Clone> Filter<[T]> for Simple<'a, T> {
    fn filter_indices(&mut self, input: &[T]) -> Vec<usize> {
        let mut indices = (0..input.len()).collect::<Vec<_>>();
        indices.retain(|&index| (self.constraint)(&input[index]));
        indices
    }

    fn filter_all_indices(&mut self, input: &[T]) -> (Vec<usize>, Vec<usize>) {
        let mut indices = (0..input.len()).collect::<Vec<_>>();
        let mut removed = Vec::with_capacity(indices.len());
        indices.retain(|&index| {
            let ret = (self.constraint)(&input[index]);
            if !ret {
                removed.push(index)
            }
            ret
        });
        (indices, removed)
    }
}

impl<'a, T: ComplexField<RealField = T>, I: std::fmt::Debug + Clone>
    ApproxFilter<PointCloud<Point3Infoed<T, I>>> for Simple<'a, Point3Infoed<T, I>>
{
    fn filter(&mut self, input: &PointCloud<Point3Infoed<T, I>>) -> PointCloud<Point3Infoed<T, I>> {
        let mut storage = Vec::from(&**input);
        storage.retain(|point| (self.constraint)(point));
        PointCloud::from_vec(storage, 1)
    }
}
