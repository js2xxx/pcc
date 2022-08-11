#![feature(map_try_insert)]

mod bilateral;
pub mod convolution;
mod crop;
mod local_max;
mod frustum;
mod inlier_proj;
mod median;
mod outlier_removal;
mod random;
mod simple;
mod uniform_sa;
mod voxel_grid;

pub use self::{
    bilateral::Bilateral,
    crop::CropBox,
    local_max::LocalMaximumZ,
    frustum::FrustumCulling,
    inlier_proj::InlierProjection,
    median::Median2,
    outlier_removal::{RadiusOutlierRemoval, StatOutlierRemoval},
    random::Random,
    simple::Simple,
    uniform_sa::UniformSampling,
    voxel_grid::{HashVoxelGrid, VoxelGrid},
};
