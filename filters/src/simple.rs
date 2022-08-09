use pcc_common::{filter::Filter, point_cloud::PointCloud};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Simple<'a, T> {
    indices: Vec<usize>,
    point_cloud: &'a PointCloud<T>,
}

impl<'a, T> Simple<'a, T> {
    pub fn new(point_cloud: &'a PointCloud<T>) -> Self {
        Simple {
            indices: (0..point_cloud.len()).collect(),
            point_cloud,
        }
    }
}

impl<'a, T> Filter<'a, T> for Simple<'a, T> {
    fn append_constraint<C>(&mut self, mut constraint: C)
    where
        C: FnMut(&'a T) -> bool,
    {
        self.indices
            .retain(|&index| constraint(&(*self.point_cloud)[index]))
    }

    fn result(&self) -> &[usize] {
        &self.indices
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimpleWithRemoved<'a, T> {
    indices: Vec<usize>,
    removed: Vec<usize>,
    point_cloud: &'a PointCloud<T>,
}

impl<'a, T> SimpleWithRemoved<'a, T> {
    pub fn new(point_cloud: &'a PointCloud<T>) -> Self {
        SimpleWithRemoved {
            indices: (0..point_cloud.len()).collect(),
            removed: Vec::with_capacity(point_cloud.len()),
            point_cloud,
        }
    }
}

impl<'a, T> Filter<'a, T> for SimpleWithRemoved<'a, T> {
    fn append_constraint<C>(&mut self, mut constraint: C)
    where
        C: FnMut(&'a T) -> bool,
    {
        self.indices.retain(|&index| {
            let ret = constraint(&(*self.point_cloud)[index]);
            if ret {
                self.removed.push(index)
            }
            ret
        })
    }

    fn result(&self) -> &[usize] {
        &self.indices
    }

    fn removed(&self) -> &[usize] {
        &self.removed
    }
}
