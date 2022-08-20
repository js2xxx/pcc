mod convert;
mod parse;

use std::{error::Error, io::BufRead};

use nalgebra::ComplexField;
use pcc_common::{
    point::{Point, PointFields},
    point_cloud::PointCloud,
};

pub use self::{convert::Viewpoint, parse::*};

pub struct Pcd {
    pub header: PcdHeader,
    pub finite: bool,
    pub data: Vec<u8>,
}

impl Pcd {
    pub fn parse<R: BufRead>(mut reader: R) -> Result<Self, Box<dyn Error>> {
        let header = PcdHeader::parse(&mut reader)?;
        let mut data = Vec::new();
        let finite = header.data.parse(reader, &header.fields, &mut data)?;
        Ok(Pcd {
            header,
            finite,
            data,
        })
    }
}

#[inline]
pub fn parse_pcd<P, R: BufRead>(reader: R) -> Result<(PointCloud<P>, Viewpoint), Box<dyn Error>>
where
    P: Point + PointFields,
    P::Data: ComplexField,
{
    let pcd = Pcd::parse(reader)?;
    pcd.to_point_cloud()
}
