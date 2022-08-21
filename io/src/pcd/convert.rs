use core::slice;
use std::{error::Error, mem};

use nalgebra::{ComplexField, Quaternion, Vector3};
use num::{FromPrimitive, One};
use pcc_common::{
    point::{Point, PointFields},
    point_cloud::PointCloud,
};

use super::{Pcd, PcdData, PcdField, PcdFieldData, PcdFieldType, PcdHeader};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Viewpoint {
    pub origin: Vector3<f32>,
    pub quat: Quaternion<f32>,
}

impl Pcd {
    pub fn from_point_cloud<P>(
        point_cloud: &PointCloud<P>,
        viewpoint: &Viewpoint,
        data_type: PcdData,
    ) -> Self
    where
        P: Point + PointFields,
        P::Data: PcdFieldData,
    {
        let fields = <P as PointFields>::fields();
        let pcd_fields = { fields.clone() }
            .map(PcdField::from_info::<P::Data>)
            .collect::<Vec<_>>();

        let rec_size =
            { pcd_fields.iter() }.fold(0, |acc, field| acc + field.count * field.ty.size());

        let header = PcdHeader {
            fields: pcd_fields,
            rec_size,
            width: point_cloud.width(),
            height: point_cloud.height(),
            viewpoint_origin: viewpoint.origin,
            viewpoint_quat: viewpoint.quat,
            data: data_type,
        };

        let mut data = Vec::with_capacity(rec_size * header.width * header.height);
        for point in point_cloud.iter() {
            let src_slice = point.as_slice();
            for field in fields.clone() {
                let field_size = field.len * mem::size_of::<P::Data>();
                let src = {
                    let src = &src_slice[field.offset..][..field.len];
                    unsafe { slice::from_raw_parts(src.as_ptr() as *const u8, field_size) }
                };
                data.extend_from_slice(src);
            }
        }

        Pcd {
            header,
            finite: point_cloud.is_bounded(),
            data,
        }
    }

    pub fn to_point_cloud<P>(self) -> Result<(PointCloud<P>, Viewpoint), Box<dyn Error>>
    where
        P: Point + PointFields,
        P::Data: ComplexField,
    {
        let fields = {
            let mut fields = <P as PointFields>::fields()
                .map(|field| (field, None))
                .collect::<Vec<_>>();
            fields.sort_by_key(|(field, _)| field.name);
            for pcd_field in self.header.fields {
                let entry = match &*pcd_field.name {
                    "rgb" => fields.binary_search_by_key(&"rgba", |(field, _)| field.name),
                    name => fields.binary_search_by_key(&name, |(field, _)| field.name),
                };
                if let Ok((_, pcds)) = entry.map(|index| &mut fields[index]) {
                    if let Some(old) = pcds.replace(pcd_field) {
                        return Err(format!(
                            "Found multiple fields in PCD file matching one field in the point cloud: {:?}", 
                            old
                        ).into());
                    }
                }
            }
            if fields.iter().any(|(_, pcd)| pcd.is_none()) {
                log::warn!(
                    "Found a field in the point cloud with no matching field in the PCD file, 
keeping with default values"
                )
            }
            fields.sort_by_key(|(field, _)| field.offset);
            fields
        };

        let mut storage = vec![P::default(); self.header.width * self.header.height];
        for (src, dst) in { self.data.chunks(self.header.rec_size) }.zip(storage.iter_mut()) {
            let mut pcd_offset = 0;
            let dst_slice = dst.as_mut_slice();

            for (field, pcd_field) in
                { fields.iter() }.map(|(field, pcd_field)| (field, pcd_field.as_ref().unwrap()))
            {
                let dst = &mut dst_slice[field.offset..][..field.len];
                let src = &src[pcd_offset..][..(pcd_field.ty.size() * pcd_field.count)];
                match pcd_field.ty {
                    PcdFieldType::U8 => {
                        for (src, dst) in src.iter().zip(dst.iter_mut()) {
                            *dst = P::Data::from_u8(*src).unwrap();
                        }
                    }
                    PcdFieldType::I8 => {
                        for (src, dst) in src.iter().zip(dst.iter_mut()) {
                            *dst = P::Data::from_i8(*src as i8).unwrap();
                        }
                    }
                    PcdFieldType::U16 => {
                        for (src, dst) in src.chunks(2).zip(dst.iter_mut()) {
                            *dst = P::Data::from_u16(u16::from_ne_bytes(src.try_into().unwrap()))
                                .unwrap();
                        }
                    }
                    PcdFieldType::I16 => {
                        for (src, dst) in src.chunks(2).zip(dst.iter_mut()) {
                            *dst = P::Data::from_i16(i16::from_ne_bytes(src.try_into().unwrap()))
                                .unwrap();
                        }
                    }
                    PcdFieldType::U32 => {
                        for (src, dst) in src.chunks(4).zip(dst.iter_mut()) {
                            *dst = P::Data::from_u32(u32::from_ne_bytes(src.try_into().unwrap()))
                                .unwrap();
                        }
                    }
                    PcdFieldType::I32 => {
                        for (src, dst) in src.chunks(4).zip(dst.iter_mut()) {
                            *dst = P::Data::from_i32(i32::from_ne_bytes(src.try_into().unwrap()))
                                .unwrap();
                        }
                    }
                    PcdFieldType::F32 => {
                        for (src, dst) in src.chunks(4).zip(dst.iter_mut()) {
                            *dst = P::Data::from_f32(f32::from_ne_bytes(src.try_into().unwrap()))
                                .unwrap();
                        }
                    }
                    PcdFieldType::U64 => {
                        for (src, dst) in src.chunks(8).zip(dst.iter_mut()) {
                            *dst = P::Data::from_u64(u64::from_ne_bytes(src.try_into().unwrap()))
                                .unwrap();
                        }
                    }
                    PcdFieldType::I64 => {
                        for (src, dst) in src.chunks(8).zip(dst.iter_mut()) {
                            *dst = P::Data::from_i64(i64::from_ne_bytes(src.try_into().unwrap()))
                                .unwrap();
                        }
                    }
                    PcdFieldType::F64 => {
                        for (src, dst) in src.chunks(8).zip(dst.iter_mut()) {
                            *dst = P::Data::from_f64(f64::from_ne_bytes(src.try_into().unwrap()))
                                .unwrap();
                        }
                    }
                    PcdFieldType::U128 => {
                        for (src, dst) in src.chunks(16).zip(dst.iter_mut()) {
                            *dst = P::Data::from_u128(u128::from_ne_bytes(src.try_into().unwrap()))
                                .unwrap();
                        }
                    }
                    PcdFieldType::I128 => {
                        for (src, dst) in src.chunks(16).zip(dst.iter_mut()) {
                            *dst = P::Data::from_i128(i128::from_ne_bytes(src.try_into().unwrap()))
                                .unwrap();
                        }
                    }
                }
                pcd_offset += src.len();
            }

            dst.coords_mut().w = P::Data::one();
        }

        let point_cloud =
            unsafe { PointCloud::from_raw_parts(storage, self.header.width, self.finite) };
        let viewpoint = Viewpoint {
            origin: self.header.viewpoint_origin,
            quat: self.header.viewpoint_quat,
        };
        Ok((point_cloud, viewpoint))
    }
}

impl<P> TryFrom<Pcd> for (PointCloud<P>, Viewpoint)
where
    P: Point + PointFields,
    P::Data: ComplexField,
{
    type Error = Box<dyn Error>;

    fn try_from(value: Pcd) -> Result<Self, Self::Error> {
        value.to_point_cloud()
    }
}
