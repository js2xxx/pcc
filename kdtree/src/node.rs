use std::ptr::NonNull;

use bitvec::vec::BitVec;
use nalgebra::{ComplexField, Scalar, Vector4};

use crate::ResultSet;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum Node<'a, T: Scalar> {
    Leaf {
        index: usize,
        coord: &'a Vector4<T>,
    },
    Branch {
        children: [NonNull<Node<'a, T>>; 2],
        dim: usize,
        value: T,
    },
}

impl<'a, T: Scalar> Node<'a, T> {
    pub(crate) fn new_leaf(index: usize, coord: &'a Vector4<T>) -> Self {
        Node::Leaf { index, coord }
    }

    /// # Safety
    /// 
    /// The caller must not use the data in the node after calling this function.
    pub(crate) unsafe fn destroy(&mut self) {
        match self {
            Node::Leaf { .. } => {}
            Node::Branch {
                children: [left, right],
                ..
            } => {
                left.as_mut().destroy();
                right.as_mut().destroy();

                let _ = (Box::from_raw(left.as_ptr()), Box::from_raw(right.as_ptr()));
            },
        }
    }
}

impl<'a, T: Scalar + Copy + ComplexField<RealField = T> + PartialOrd> Node<'a, T> {
    pub fn insert(&mut self, index: usize, pivot: &'a Vector4<T>) {
        let mut node = self;
        loop {
            let mut next = match *node {
                Node::Leaf {
                    index: one_index,
                    coord,
                } => {
                    let (dim, _) = { pivot.xyz().iter() }
                        .zip(coord.xyz().iter())
                        .map(|(&x, &y)| (x - y).abs())
                        .enumerate()
                        .fold(
                            (0, T::zero()),
                            |(max_dim, max_distance), (dim, distance)| {
                                if distance > max_distance {
                                    (dim, distance)
                                } else {
                                    (max_dim, max_distance)
                                }
                            },
                        );
                    let one = Box::leak(Box::new(Node::new_leaf(one_index, coord))).into();
                    let other = Box::leak(Box::new(Node::new_leaf(index, pivot))).into();

                    *node = Node::Branch {
                        children: if coord[dim] < pivot[dim] {
                            [one, other]
                        } else {
                            [other, one]
                        },
                        dim,
                        value: (coord[dim] + pivot[dim]) / (T::one() + T::one()),
                    };

                    break;
                }
                Node::Branch {
                    children: [left, right],
                    dim,
                    value,
                } => {
                    if pivot[dim] < value {
                        left
                    } else {
                        right
                    }
                }
            };

            node = unsafe { next.as_mut() }
        }
    }
}

fn check_and_set(index: usize, checker: &mut BitVec) -> bool {
    let ret = matches!(checker.get(index), Some(c) if *c);
    if !ret {
        if checker.len() <= index {
            checker.resize(index + 1, false);
        }
        checker.set(index, true);
    }
    ret
}

impl<'a, T: Scalar + Copy + ComplexField<RealField = T> + Ord> Node<'a, T> {
    fn search_one(
        &self,
        pivot: &Vector4<T>,
        result: &mut impl ResultSet<T, &'a Vector4<T>>,
        other_branches: &mut Vec<NonNull<Node<'a, T>>>,
        checker: &mut BitVec,
    ) {
        let mut node = self;
        loop {
            match *node {
                Node::Leaf { index, coord } => {
                    if !check_and_set(index, checker) {
                        let distance = (coord.xyz() - pivot.xyz()).norm();
                        result.push(distance, coord);
                    }
                    break;
                }
                Node::Branch {
                    children: [left, right],
                    dim,
                    value,
                } => {
                    let (next, other) = if pivot[dim] < value {
                        (left, Some(right))
                    } else {
                        (right, Some(left))
                    };

                    let min_distance = (pivot[dim] - value).abs();
                    if let Some(other) = other {
                        if result.max_key() < Some(&min_distance) || !result.is_full() {
                            other_branches.push(other)
                        }
                    }

                    node = unsafe { next.as_ref() }
                }
            }
        }
    }

    pub fn search_exact(&self, pivot: &Vector4<T>, result: &mut impl ResultSet<T, &'a Vector4<T>>) {
        match *self {
            Node::Leaf { coord, .. } => {
                let distance = (coord.xyz() - pivot.xyz()).norm();
                result.push(distance, coord);
            }
            Node::Branch {
                children: [left, right],
                dim,
                value,
            } => {
                let (next, other) = if pivot[dim] < value {
                    (left, Some(right))
                } else {
                    (right, Some(left))
                };

                unsafe { next.as_ref() }.search_exact(pivot, result);

                let min_distance = (pivot[dim] - value).abs();
                if let Some(other) = other {
                    if result.max_key() < Some(&min_distance) {
                        unsafe { other.as_ref() }.search_exact(pivot, result)
                    }
                }
            }
        }
    }

    pub fn search(&self, pivot: &Vector4<T>, result: &mut impl ResultSet<T, &'a Vector4<T>>) {
        let mut other_branches = Vec::new();
        let mut checker = BitVec::new();

        let mut node = self;
        loop {
            node.search_one(pivot, result, &mut other_branches, &mut checker);

            node = match other_branches.pop() {
                Some(node) => unsafe { node.as_ref() },
                None => break,
            }
        }
    }
}
