#![feature(array_methods)]
#![feature(array_windows)]

mod border;
mod boundary;
mod fpfh;
mod narf;
mod normal;
mod pfh;
mod vfh;

pub use self::{
    border::{BorderEstimation, BorderTraits},
    boundary::BoundaryEstimation,
    fpfh::FpfhEstimation,
    narf::{Narf, SurfacePatch},
    normal::NormalEstimation,
    pfh::PfhEstimation,
    vfh::VfhEstimation,
};

pub const HIST_MAX: f64 = 100.;
