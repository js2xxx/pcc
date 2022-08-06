#![feature(array_try_from_fn)]
#![feature(box_into_inner)]

mod base;
mod node;
mod search;

pub use base::OcTree;
pub use search::*;