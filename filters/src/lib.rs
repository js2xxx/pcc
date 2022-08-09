#![feature(map_try_insert)]

mod crop_box;
mod median;
mod outlier_removal;
mod simple;
mod voxel_grid;

pub use self::{
    crop_box::CropBox,
    median::Median2,
    outlier_removal::{RadiusOutlierRemoval, StatOutlierRemoval},
    simple::Simple,
    voxel_grid::{HashVoxelGrid, VoxelGrid},
};
