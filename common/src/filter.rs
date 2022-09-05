use crate::{point::Data, point_cloud::PointCloud};

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

    #[inline]
    fn filter_mut(&mut self, obj: &mut T) {
        *obj = self.filter(obj);
    }
}

impl<P: Data> ApproxFilter<PointCloud<P>> for [usize] {
    #[inline]
    fn filter(&mut self, input: &PointCloud<P>) -> PointCloud<P> {
        input.create_sub(self, 1)
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

impl<P: Data, F> ApproxFilter<PointCloud<P>> for F
where
    F: FnMut(&P) -> bool,
{
    fn filter(&mut self, input: &PointCloud<P>) -> PointCloud<P> {
        let mut storage = Vec::from(&**input);
        storage.retain(|point| (self)(point));
        PointCloud::from_vec(storage, 1)
    }

    fn filter_mut(&mut self, obj: &mut PointCloud<P>) {
        let storage = unsafe { obj.storage() };
        storage.retain(|point| (self)(point));
        obj.reinterpret(1);
    }
}
