#![feature(map_try_insert)]

mod simple;
mod voxel_grid;

pub use self::{
    simple::Simple,
    voxel_grid::{HashVoxelGrid, VoxelGrid},
};
