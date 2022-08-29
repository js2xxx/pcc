use std::fmt::Debug;

use nalgebra::{matrix, RealField, Scalar};
use num::ToPrimitive;
use pcc_common::{
    filter::{ApproxFilter, Filter},
    point::Point,
    point_cloud::PointCloud,
};
use pcc_sac::Plane;
use pcc_search::{KdTree, RadiusResultSet};

use crate::InlierProjection;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LocalMaximumZ<T: Scalar> {
    pub radius: T,
}

impl<T: Scalar> LocalMaximumZ<T> {
    pub fn new(radius: T) -> Self {
        LocalMaximumZ { radius }
    }
}

impl<T: RealField + ToPrimitive> LocalMaximumZ<T> {
    fn projected<P: Point<Data = T>>(&self, input: &PointCloud<P>) -> PointCloud<P> {
        let mut projector = InlierProjection::new(
            Plane {
                coords: matrix![T::zero(); T::zero(); T::one(); T::one()],
                normal: matrix![T::zero(); T::zero(); T::one(); T::zero()],
            },
            (0..input.len()).collect(),
        );
        projector.filter(input)
    }

    fn filter_inner<P: Point<Data = T>, U>(
        &self,
        input: &PointCloud<P>,
        retainer: &mut Vec<U>,
        removed: Option<&mut Vec<usize>>,
    ) {
        macro_rules! retain {
            ($condition:expr) => {
                match removed {
                    Some(removed) => retainer.retain(|_| {
                        let (index, ret) = $condition();
                        if !ret {
                            removed.push(index)
                        }
                        ret
                    }),
                    None => retainer.retain(|_| $condition().1),
                }
            };
        }
        let projected = self.projected(input);

        let searcher = KdTree::new(input);
        let mut result = RadiusResultSet::new(self.radius.clone());
        let mut visited = vec![false; input.len()];

        let mut index = 0;
        let mut condition = || {
            if visited[index] {
                index += 1;
                return (index - 1, true);
            }

            searcher.search_typed(projected[index].coords(), &mut result);
            let iter = result.iter();
            let ret =
                { iter.clone() }.any(|(_, i)| input[*i].coords().z >= input[index].coords().z);
            if !ret {
                for (_, i) in iter {
                    visited[*i] = true;
                }
            }
            index += 1;
            (index - 1, ret)
        };
        retain!(condition)
    }
}

impl<T: RealField + ToPrimitive, P: Point<Data = T>> Filter<PointCloud<P>> for LocalMaximumZ<T> {
    fn filter_indices(&mut self, input: &PointCloud<P>) -> Vec<usize> {
        let mut indices = (0..input.len()).collect::<Vec<_>>();
        self.filter_inner(input, &mut indices, None);

        indices
    }

    fn filter_all_indices(&mut self, input: &PointCloud<P>) -> (Vec<usize>, Vec<usize>) {
        let mut indices = (0..input.len()).collect::<Vec<_>>();
        let mut removed = Vec::with_capacity(indices.len());
        self.filter_inner(input, &mut indices, Some(&mut removed));

        (indices, removed)
    }
}

impl<T: RealField + ToPrimitive, P: Point<Data = T>> ApproxFilter<PointCloud<P>>
    for LocalMaximumZ<T>
{
    fn filter(&mut self, input: &PointCloud<P>) -> PointCloud<P> {
        let mut storage = Vec::from(&**input);
        self.filter_inner(input, &mut storage, None);

        PointCloud::from_vec(storage, 1)
    }
}
