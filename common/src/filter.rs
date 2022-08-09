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

impl<T: ComplexField<RealField = T>, I: std::fmt::Debug + Clone>
    ApproxFilter<PointCloud<Point3Infoed<T, I>>> for [usize]
{
    fn filter(&mut self, input: &PointCloud<Point3Infoed<T, I>>) -> PointCloud<Point3Infoed<T, I>> {
        PointCloud::from_vec(self.iter().map(|&index| input[index].clone()).collect(), 1)
    }
}
