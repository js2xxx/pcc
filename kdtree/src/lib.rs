#![feature(type_alias_impl_trait)]

mod node;
mod result;

use std::ptr::NonNull;

use nalgebra::{RealField, Scalar, Vector4};
use node::Node;
use pcc_common::{point_cloud::PointCloud, points::Point3Infoed, search::SearchType};

pub use self::result::*;

pub struct KdTree<'a, T: Scalar> {
    root: Option<NonNull<Node<'a, T>>>,
    indices: Vec<usize>,
}

impl<'a, T: Scalar> Default for KdTree<'a, T> {
    fn default() -> Self {
        KdTree {
            root: None,
            indices: Vec::new(),
        }
    }
}

impl<'a, T: Copy + RealField> KdTree<'a, T> {
    pub fn insert(&mut self, index: usize, pivot: &'a Vector4<T>) {
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

impl<'a, T: Copy + RealField> KdTree<'a, T> {
    pub fn search(&self, pivot: &Vector4<T>, result: &mut impl ResultSet<Key = T, Value = usize>) {
        if let Some(root) = self.root {
            unsafe { root.as_ref() }.search(pivot, result)
        }
    }

    pub fn search_exact(
        &self,
        pivot: &Vector4<T>,
        result: &mut impl ResultSet<Key = T, Value = usize>,
    ) {
        if let Some(root) = self.root {
            unsafe { root.as_ref() }.search_exact(pivot, result)
        }
    }
}

impl<'a, T: Scalar> Drop for KdTree<'a, T> {
    fn drop(&mut self) {
        if let Some(mut root) = self.root {
            unsafe {
                root.as_mut().destroy();
                let _ = Box::from_raw(root.as_ptr());
            }
        }
    }
}

impl<'a, T: RealField + Copy> KdTree<'a, T> {
    pub fn new<I>(point_cloud: &'a PointCloud<Point3Infoed<T, I>>) -> Self {
        if !point_cloud.is_empty() {
            let mut indices = (0..point_cloud.len()).collect::<Vec<_>>();
            let root = Node::build(0, point_cloud, &mut indices, None);
            KdTree {
                root: Some(root),
                indices,
            }
        } else {
            Default::default()
        }
    }
}

impl<'a, T: RealField + Copy> pcc_common::search::Searcher<'a, T> for KdTree<'a, T> {
    fn search(&self, pivot: &Vector4<T>, ty: SearchType<T>, result: &mut Vec<usize>) {
        result.clear();
        match ty {
            SearchType::Knn(num) => {
                let mut rs = KnnResultSet::new(num);
                self.search(pivot, &mut rs);
                result.extend(rs.into_iter().map(|(_, v)| v));
            }
            SearchType::Radius(radius) => {
                let mut rs = RadiusResultSet::new(radius);
                self.search(pivot, &mut rs);
                result.extend(rs.into_iter().map(|(_, v)| v));
            }
        }
    }

    fn search_exact(&self, pivot: &Vector4<T>, ty: SearchType<T>, result: &mut Vec<usize>) {
        result.clear();
        match ty {
            SearchType::Knn(num) => {
                let mut rs = KnnResultSet::new(num);
                self.search_exact(pivot, &mut rs);
                result.extend(rs.into_iter().map(|(_, v)| v));
            }
            SearchType::Radius(radius) => {
                let mut rs = RadiusResultSet::new(radius);
                self.search_exact(pivot, &mut rs);
                result.extend(rs.into_iter().map(|(_, v)| v));
            }
        }
    }
}
