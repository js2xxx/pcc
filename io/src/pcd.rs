mod parse;

use std::{error::Error, io::BufRead};

pub use self::parse::*;

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
