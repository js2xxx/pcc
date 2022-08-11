#![feature(map_try_insert)]

mod bilateral;
pub mod convolution;
mod crop;
mod frustum;
mod inlier_proj;
mod local_max;
mod median;
mod outlier_removal;
mod random;
mod simple;
mod uniform_sa;
mod voxel_grid;

pub use self::{
    bilateral::Bilateral,
    crop::CropBox,
    frustum::FrustumCulling,
    inlier_proj::InlierProjection,
    local_max::LocalMaximumZ,
    median::Median2,
    outlier_removal::{RadiusOutlierRemoval, StatOutlierRemoval},
    random::Random,
    simple::Simple,
    uniform_sa::UniformSampling,
    voxel_grid::{GridMinimumZ, HashVoxelGrid, VoxelGrid},
};
