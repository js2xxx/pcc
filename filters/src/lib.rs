#![feature(map_try_insert)]

mod crop_box;
mod inlier_proj;
mod median;
mod outlier_removal;
mod random;
mod simple;
mod uniform_sa;
mod voxel_grid;

pub use self::{
    crop_box::CropBox,
    inlier_proj::InlierProjection,
    median::Median2,
    outlier_removal::{RadiusOutlierRemoval, StatOutlierRemoval},
    random::Random,
    simple::Simple,
    uniform_sa::UniformSampling,
    voxel_grid::{HashVoxelGrid, VoxelGrid},
};
