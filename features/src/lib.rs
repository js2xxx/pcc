#![feature(array_windows)]

mod boundary;
mod normal;

pub use self::{boundary::BoundaryEstimation, normal::NormalEstimation};
