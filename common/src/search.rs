use nalgebra::{Scalar, Vector4};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SearchType<T> {
    Knn(usize),
    Radius(T),
}

pub trait Searcher<'a, T: Scalar> {
    fn search(&self, pivot: &Vector4<T>, ty: SearchType<T>, result: &mut Vec<usize>);

    fn search_exact(&self, pivot: &Vector4<T>, ty: SearchType<T>, result: &mut Vec<usize>) {
        self.search(pivot, ty, result)
    }
}
