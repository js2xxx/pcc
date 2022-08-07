#![feature(array_try_from_fn)]
#![feature(box_into_inner)]

mod adjacency;
mod base;
mod centroid;
mod count;
mod iter;
mod node;
mod point_cloud;
mod search;

pub use self::{
    adjacency::OcTreePcAdjacency,
    base::OcTree,
    centroid::OcTreePcCentroid,
    count::OcTreePcCount,
    iter::{DepthIter, DepthIterMut},
    point_cloud::{CreateOptions, OcTreePc},
    search::OcTreePcSearch,
};
