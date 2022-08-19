#![feature(type_alias_impl_trait)]

mod node;
mod result;

use std::ptr::NonNull;

use nalgebra::{RealField, Vector4};
use node::Node;
use pcc_common::{point::Point, point_cloud::PointCloud, search::SearchType};

pub use self::result::*;

pub struct KdTree<'a, P: Point> {
    point_cloud: &'a PointCloud<P>,
    root: Option<NonNull<Node<'a, P::Data>>>,
    indices: Vec<usize>,
}

unsafe impl<'a, P: Point + Send> Send for KdTree<'a, P> {}
unsafe impl<'a, P: Point + Sync> Sync for KdTree<'a, P> {}

impl<'a, P: Point> KdTree<'a, P>
where
    P::Data: RealField,
{
    pub fn insert(&mut self, index: usize, pivot: &'a Vector4<P::Data>) {
        match self.root {
            Some(mut root) => unsafe { root.as_mut() }.insert(index, pivot),
            None => {
                let node = Box::leak(Box::new(Node::new_leaf(index, pivot)));
                self.root = Some(node.into());
            }
        }
        if self.indices.len() <= index {
            self.indices.resize(index + 1, 0)
        }
        self.indices[index] = index;
    }
}

impl<'a, P: Point> KdTree<'a, P>
where
    P::Data: RealField,
{
    pub fn search_typed(
        &self,
        pivot: &Vector4<P::Data>,
        result: &mut impl ResultSet<Key = P::Data, Value = usize>,
    ) {
        if let Some(root) = self.root {
            unsafe { root.as_ref() }.search(pivot, result)
        }
    }

    pub fn search_exact_typed(
        &self,
        pivot: &Vector4<P::Data>,
        result: &mut impl ResultSet<Key = P::Data, Value = usize>,
    ) {
        if let Some(root) = self.root {
            unsafe { root.as_ref() }.search_exact(pivot, result)
        }
    }
}

impl<'a, P: Point> Drop for KdTree<'a, P> {
    fn drop(&mut self) {
        if let Some(mut root) = self.root {
            unsafe {
                root.as_mut().destroy();
                let _ = Box::from_raw(root.as_ptr());
            }
        }
    }
}

impl<'a, P: Point> KdTree<'a, P>
where
    P::Data: RealField,
{
    pub fn new(point_cloud: &'a PointCloud<P>) -> Self {
        assert!(!point_cloud.is_empty());

        let mut indices = (0..point_cloud.len()).collect::<Vec<_>>();
        let root = Node::build(0, point_cloud, &mut indices, None);
        KdTree {
            point_cloud,
            root: Some(root),
            indices,
        }
    }
}

impl<'a, P: Point> pcc_common::search::Searcher<'a, P> for KdTree<'a, P>
where
    P::Data: RealField,
{
    fn point_cloud(&self) -> &'a PointCloud<P> {
        self.point_cloud
    }

    fn search(
        &self,
        pivot: &Vector4<P::Data>,
        ty: SearchType<P::Data>,
        result: &mut Vec<(usize, P::Data)>,
    ) {
        result.clear();
        match ty {
            SearchType::Knn(num) => {
                let mut rs = KnnResultSet::new(num);
                self.search_typed(pivot, &mut rs);
                result.extend(rs.into_iter().map(|(d, v)| (v, d)));
            }
            SearchType::Radius(radius) => {
                let mut rs = RadiusResultSet::new(radius);
                self.search_typed(pivot, &mut rs);
                result.extend(rs.into_iter().map(|(d, v)| (v, d)));
            }
        }
    }

    fn search_exact(
        &self,
        pivot: &Vector4<P::Data>,
        ty: SearchType<P::Data>,
        result: &mut Vec<(usize, P::Data)>,
    ) {
        result.clear();
        match ty {
            SearchType::Knn(num) => {
                let mut rs = KnnResultSet::new(num);
                self.search_exact_typed(pivot, &mut rs);
                result.extend(rs.into_iter().map(|(d, v)| (v, d)));
            }
            SearchType::Radius(radius) => {
                let mut rs = RadiusResultSet::new(radius);
                self.search_exact_typed(pivot, &mut rs);
                result.extend(rs.into_iter().map(|(d, v)| (v, d)));
            }
        }
    }
}
