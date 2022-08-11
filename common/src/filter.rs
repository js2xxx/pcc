use nalgebra::ComplexField;

use crate::{point_cloud::PointCloud, points::Point3Infoed};

/// A filter that keeps some parts of input, for example, some elements of an
/// array, and transfers them to the output.
pub trait Filter<T: ?Sized> {
    /// Thus, the array output are not necessary at all times, but the indices
    /// in the input is returned in order to reduce the memory usage.
    fn filter_indices(&mut self, input: &T) -> Vec<usize>;

    /// This function may return less than the exact result of the removed
    /// indices of points. The empty slices returned are often considered not
    /// stored by the filter.
    fn filter_all_indices(&mut self, input: &T) -> (Vec<usize>, Vec<usize>) {
        (self.filter_indices(input), Vec::new())
    }
}

/// A filter that often generate an approximation of some parts of input,
/// instead of keeping all parts consistent.
pub trait ApproxFilter<T> {
    fn filter(&mut self, input: &T) -> T;
}

impl<T: ComplexField, I: std::fmt::Debug + Clone> ApproxFilter<PointCloud<Point3Infoed<T, I>>>
    for [usize]
{
    fn filter(&mut self, input: &PointCloud<Point3Infoed<T, I>>) -> PointCloud<Point3Infoed<T, I>> {
        PointCloud::from_vec(self.iter().map(|&index| input[index].clone()).collect(), 1)
    }
}

impl<T, F: FnMut(&T) -> bool> Filter<[T]> for F {
    fn filter_indices(&mut self, input: &[T]) -> Vec<usize> {
        let mut indices = (0..input.len()).collect::<Vec<_>>();
        indices.retain(|&index| (self)(&input[index]));
        indices
    }

    fn filter_all_indices(&mut self, input: &[T]) -> (Vec<usize>, Vec<usize>) {
        let mut indices = (0..input.len()).collect::<Vec<_>>();
        let mut removed = Vec::with_capacity(indices.len());
        indices.retain(|&index| {
            let ret = (self)(&input[index]);
            if !ret {
                removed.push(index)
            }
            ret
        });
        (indices, removed)
    }
}

impl<T: ComplexField, I: std::fmt::Debug + Clone, F> ApproxFilter<PointCloud<Point3Infoed<T, I>>>
    for F
where
    F: FnMut(&Point3Infoed<T, I>) -> bool,
{
    fn filter(&mut self, input: &PointCloud<Point3Infoed<T, I>>) -> PointCloud<Point3Infoed<T, I>> {
        let mut storage = Vec::from(&**input);
        storage.retain(|point| (self)(point));
        PointCloud::from_vec(storage, 1)
    }
}
