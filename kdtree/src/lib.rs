#![feature(type_alias_impl_trait)]

mod node;
mod result;

use std::ptr::NonNull;

use nalgebra::{ComplexField, Scalar, Vector4};
use node::Node;

pub use self::result::*;

pub struct KdTree<'a, T: Scalar> {
    root: Option<NonNull<Node<'a, T>>>,
}

impl<'a, T: Scalar> KdTree<'a, T> {
    pub fn new() -> Self {
        KdTree { root: None }
    }
}

impl<'a, T: Scalar + Copy + ComplexField<RealField = T> + PartialOrd> KdTree<'a, T> {
    pub fn insert(&mut self, index: usize, pivot: &'a Vector4<T>) {
        match self.root {
            Some(mut root) => unsafe { root.as_mut() }.insert(index, pivot),
            None => {
                let node = Box::leak(Box::new(Node::new_leaf(index, pivot)));
                self.root = Some(node.into());
            }
        }
    }
}

impl<'a, T: Scalar + Copy + ComplexField<RealField = T> + Ord> KdTree<'a, T> {
    pub fn search(&self, pivot: &Vector4<T>, result: &mut impl ResultSet<T, &'a Vector4<T>>) {
        if let Some(root) = self.root {
            unsafe { root.as_ref() }.search(pivot, result)
        }
    }

    pub fn search_exact(&self, pivot: &Vector4<T>, result: &mut impl ResultSet<T, &'a Vector4<T>>) {
        if let Some(root) = self.root {
            unsafe { root.as_ref() }.search_exact(pivot, result)
        }
    }
}

impl<'a, T: Scalar> Default for KdTree<'a, T> {
    fn default() -> Self {
        Self::new()
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
