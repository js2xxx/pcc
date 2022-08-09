use std::ptr::NonNull;

use bitvec::vec::BitVec;
use nalgebra::{RealField, Scalar, Vector3, Vector4};

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
    /// The caller must not use the data in the node after calling this
    /// function.
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
            }
        }
    }
}

fn cut_split<T: Scalar + PartialOrd, P: AsRef<Vector4<T>>>(
    coords: &[P],
    indices: &mut [usize],
    dim: usize,
    value: T,
) -> (usize, usize) {
    let mut left = 0;
    let mut right = indices.len() - 1;
    loop {
        while left <= right && coords[indices[left]].as_ref()[dim] < value {
            left += 1
        }
        while left <= right && coords[indices[right]].as_ref()[dim] >= value {
            right -= 1
        }
        if left > right {
            break;
        }
        indices.swap(left, right);
        left += 1;
        right -= 1;
    }

    let limit_left = left;
    right = indices.len() - 1;
    loop {
        while left <= right && coords[indices[left]].as_ref()[dim] <= value {
            left += 1
        }
        while left <= right && coords[indices[right]].as_ref()[dim] > value {
            right -= 1
        }
        if left > right {
            break;
        }
        indices.swap(left, right);
        left += 1;
        right -= 1;
    }

    let limit_right = left;

    (limit_left, limit_right)
}

fn cut<T: RealField, P: AsRef<Vector4<T>>>(
    coords: &[P],
    indices: &mut [usize],
    last: Option<usize>,
) -> (usize, usize, T) {
    let sum = { indices.iter() }
        .map(|&i| coords[i].as_ref().xyz())
        .fold(Vector3::zeros(), |acc, coord| acc + coord);

    let mean = sum / T::from_usize(coords.len()).unwrap();
    let var = { indices.iter() }.map(|&i| coords[i].as_ref().xyz()).fold(
        Vector3::zeros(),
        |acc, coord| {
            let diff = coord - mean.clone();
            acc + diff.component_mul(&diff)
        },
    );

    let dim = {
        let dim = var.imax();
        if Some(dim) == last {
            var.iter()
                .enumerate()
                .filter(|(i, _)| i != &dim)
                .fold(None, |acc, (i, v)| match acc {
                    Some(d) if v > &var[d] => Some(i),
                    _ => acc,
                })
                .unwrap()
        } else {
            dim
        }
    };

    let value = mean[dim].clone();
    let (limit_left, limit_right) = cut_split(coords, indices, dim, value);

    let mid = indices.len() / 2;
    let split = if limit_left > mid {
        limit_left
    } else if limit_right < mid {
        limit_right
    } else {
        mid
    };

    (split, dim, mean[dim].clone())
}

impl<'a, T: RealField> Node<'a, T> {
    pub fn build<P>(
        start_index: usize,
        coords: &'a [P],
        indices: &mut [usize],
        last_dim: Option<usize>,
    ) -> NonNull<Self>
    where
        P: AsRef<Vector4<T>>,
    {
        let node = if indices.len() == 1 {
            let coord: &'a Vector4<T> = coords[indices[0]].as_ref();
            Node::new_leaf(start_index, coord)
        } else {
            let (split, dim, value) = cut(coords, indices, last_dim);
            let (left, right) = indices.split_at_mut(split);

            let left = Node::build(start_index, coords, left, Some(dim));
            let right = Node::build(start_index + split, coords, right, Some(dim));

            Node::Branch {
                children: [left, right],
                dim,
                value,
            }
        };
        Box::leak(Box::new(node)).into()
    }
}

impl<'a, T: RealField> Node<'a, T> {
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
                        .map(|(x, y)| (x.clone() - y.clone()).abs())
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
                        value: (coord[dim].clone() + pivot[dim].clone()) / (T::one() + T::one()),
                    };

                    break;
                }
                Node::Branch {
                    children: [left, right],
                    dim,
                    ref value,
                } => {
                    if pivot[dim] < *value {
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

impl<'a, T: RealField> Node<'a, T> {
    fn search_one(
        &self,
        pivot: &Vector4<T>,
        result: &mut impl ResultSet<Key = T, Value = usize>,
        other_branches: &mut Vec<NonNull<Node<'a, T>>>,
        checker: &mut BitVec,
    ) {
        let mut node = self;
        loop {
            match *node {
                Node::Leaf { index, coord } => {
                    if !check_and_set(index, checker) {
                        let distance = (coord.xyz() - pivot.xyz()).norm();
                        result.push(distance, index);
                    }
                    break;
                }
                Node::Branch {
                    children: [left, right],
                    dim,
                    ref value,
                } => {
                    let (next, other) = if pivot[dim] < *value {
                        (left, Some(right))
                    } else {
                        (right, Some(left))
                    };

                    let min_distance = (pivot[dim].clone() - value.clone()).abs();
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

    pub fn search_exact(
        &self,
        pivot: &Vector4<T>,
        result: &mut impl ResultSet<Key = T, Value = usize>,
    ) {
        match *self {
            Node::Leaf { coord, index } => {
                let distance = (coord.xyz() - pivot.xyz()).norm();
                result.push(distance, index);
            }
            Node::Branch {
                children: [left, right],
                dim,
                ref value,
            } => {
                let (next, other) = if pivot[dim] < *value {
                    (left, Some(right))
                } else {
                    (right, Some(left))
                };

                unsafe { next.as_ref() }.search_exact(pivot, result);

                let min_distance = (pivot[dim].clone() - value.clone()).abs();
                if let Some(other) = other {
                    if result.max_key() < Some(&min_distance) {
                        unsafe { other.as_ref() }.search_exact(pivot, result)
                    }
                }
            }
        }
    }

    pub fn search(&self, pivot: &Vector4<T>, result: &mut impl ResultSet<Key = T, Value = usize>) {
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
