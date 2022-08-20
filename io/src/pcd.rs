mod parse;

use core::slice;
use std::{collections::HashMap, error::Error, io::BufRead};

use num::One;
use pcc_common::{
    point::{Point, PointFields},
    point_cloud::PointCloud,
};

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

impl<P> TryFrom<Pcd> for PointCloud<P>
where
    P: Point + PointFields,
    P::Data: One,
{
    type Error = Box<dyn Error>;

    fn try_from(pcd: Pcd) -> Result<Self, Self::Error> {
        let fields = {
            let mut fields = <P as PointFields>::fields()
                .map(|field| (field.name, (field, None)))
                .collect::<HashMap<_, _>>();
            for pcd_field in pcd.header.fields {
                let entry = match &*pcd_field.name {
                    "rgb" | "rgba" => fields.get_mut("rgb,rgba"),
                    name => fields.get_mut(name),
                };
                if let Some((_, pcds)) = entry {
                    if let Some(old) = pcds.replace(pcd_field) {
                        log::warn!("Found multiple fields in PCD file matching one field in the point cloud: {:?}", old);
                    }
                }
            }
            if fields.values().any(|(_, pcd)| pcd.is_none()) {
                log::warn!(
                    "Found a field in the point cloud with no matching field in the PCD file, 
keeping with default values"
                )
            }
            fields
        };

        let mut storage = vec![P::default(); pcd.header.width * pcd.header.height];
        for (src, dst) in pcd.data.chunks(pcd.header.rec_size).zip(storage.iter_mut()) {
            let mut pcd_offset = 0;
            for (field, pcd_field) in
                { fields.values() }.map(|(field, pcd_field)| (field, pcd_field.as_ref().unwrap()))
            {
                let field_size = pcd_field.count * pcd_field.ty.size();
                assert_eq!(field_size % field.len, 0);
                let src = unsafe {
                    let src = src.as_ptr().add(pcd_offset);
                    slice::from_raw_parts(src as *const P::Data, field_size / field.len)
                };
                let dst = &mut dst.as_mut_slice()[field.offset..][..field.len];
                dst.clone_from_slice(src);

                pcd_offset += src.len() * field.len;
            }

            dst.coords_mut().w = P::Data::one();
        }

        Ok(unsafe { PointCloud::from_raw_parts(storage, pcd.header.width, pcd.finite) })
    }
}
