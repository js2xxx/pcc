#![feature(map_try_insert)]

mod crop_box;
mod simple;
mod voxel_grid;
mod median;
mod outlier_removal;

pub use self::{
    crop_box::CropBox,
    simple::Simple,
    voxel_grid::{HashVoxelGrid, VoxelGrid},
    median::Median2,
    outlier_removal::StatOutlierRemoval,
};
