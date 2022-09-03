#![feature(array_windows)]

mod border;
mod boundary;
mod fpfh;
mod normal;
mod pfh;
mod vfh;

pub use self::{
    border::{BorderEstimation, BorderTraits},
    boundary::BoundaryEstimation,
    fpfh::FpfhEstimation,
    normal::NormalEstimation,
    pfh::PfhEstimation,
    vfh::VfhEstimation,
};

pub const HIST_MAX: f64 = 100.;
