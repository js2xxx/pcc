#![feature(array_methods)]
#![feature(array_windows)]

mod border;
mod boundary;
mod fpfh;
mod gasd;
mod intensity;
mod moment;
mod narf;
mod normal;
mod pfh;
mod vfh;

pub use self::{
    border::{Border, BorderTraits},
    boundary::Boundary,
    fpfh::Fpfh,
    gasd::{Gasd, GasdColor, GasdData, GasdOutput},
    intensity::IntensityGradient,
    moment::MomentInvariant,
    narf::{Narf, NarfData, SurfacePatch},
    normal::Normal,
    pfh::Pfh,
    vfh::Vfh,
};

pub const HIST_MAX: f64 = 100.;
