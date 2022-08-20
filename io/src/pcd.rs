mod convert;
mod read;
mod write;

use std::{any::TypeId, error::Error, io::{BufRead, Write}};

use nalgebra::{ComplexField, Quaternion, Vector3};
use pcc_common::{
    point::{Point, PointFields},
    point_cloud::PointCloud,
};

pub use self::convert::Viewpoint;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PcdField {
    pub name: String,
    pub ty: PcdFieldType,
    pub count: usize,
}

impl Default for PcdField {
    fn default() -> Self {
        Self {
            name: Default::default(),
            ty: PcdFieldType::F32,
            count: 1,
        }
    }
}

impl TryFrom<pcc_common::point::FieldInfo> for PcdField {
    type Error = TypeId;

    fn try_from(field: pcc_common::point::FieldInfo) -> Result<Self, Self::Error> {
        Ok(PcdField {
            name: field.name.to_string(),
            ty: PcdFieldType::from_type_id(field.ty).ok_or(field.ty)?,
            count: field.len,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PcdFieldType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
    U128,
    I128,
}

impl PcdFieldType {
    pub fn from_type_id(ty: TypeId) -> Option<Self> {
        Some(match ty {
            ty if ty == TypeId::of::<u8>() => Self::U8,
            ty if ty == TypeId::of::<i8>() => Self::I8,
            ty if ty == TypeId::of::<u16>() => Self::U16,
            ty if ty == TypeId::of::<i16>() => Self::I16,
            ty if ty == TypeId::of::<u32>() => Self::U32,
            ty if ty == TypeId::of::<i32>() => Self::I32,
            ty if ty == TypeId::of::<f32>() => Self::F32,
            ty if ty == TypeId::of::<u64>() => Self::U64,
            ty if ty == TypeId::of::<i64>() => Self::I64,
            ty if ty == TypeId::of::<f64>() => Self::F64,
            ty if ty == TypeId::of::<u128>() => Self::U128,
            ty if ty == TypeId::of::<i128>() => Self::I128,
            _ => return None,
        })
    }

    pub fn size(&self) -> usize {
        use PcdFieldType::*;
        match self {
            U8 | I8 => 1,
            U16 | I16 => 2,
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 => 8,
            U128 | I128 => 16,
        }
    }

    pub fn type_str(&self) -> &'static str {
        use PcdFieldType::*;
        match self {
            U8 | U16 | U32 | U64 | U128 => "U",
            I8 | I16 | I32 | I64 | I128 => "I",
            F32 | F64 => "F",
        }
    }

    fn default_sized(size: usize) -> Result<Self, String> {
        Ok(match size {
            1 => PcdFieldType::I8,
            2 => PcdFieldType::I16,
            4 => PcdFieldType::F32,
            8 => PcdFieldType::F64,
            16 => PcdFieldType::I128,
            _ => return Err(format!("Unknown SIZE: {:?}", size)),
        })
    }

    fn set_type(&mut self, ty: &str) -> Result<(), String> {
        use PcdFieldType::*;
        match (*self, ty) {
            (U8, "I") => *self = I8,
            (I8, "U") => *self = U8,
            (U16, "I") => *self = I16,
            (I16, "U") => *self = U16,
            (U32 | F32, "I") => *self = I32,
            (I32 | F32, "U") => *self = U32,
            (I32 | U32, "F") => *self = F32,
            (U64 | F64, "I") => *self = I64,
            (I64 | F64, "U") => *self = U64,
            (I64 | U64, "F") => *self = F64,
            (U128, "I") => *self = I128,
            (I128, "U") => *self = U128,
            (_, "I" | "U" | "F") => {}
            _ => return Err(format!("Unknown TYPE: {:?}", ty)),
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PcdData {
    Ascii,
    Binary,
    BinaryCompressed,
}

impl PcdData {
    pub fn type_str(&self) -> &'static str {
        match self {
            PcdData::Ascii => "ascii",
            PcdData::Binary => "binary",
            PcdData::BinaryCompressed => "binary_compressed",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcdHeader {
    pub fields: Vec<PcdField>,
    pub rec_size: usize,
    pub width: usize,
    pub height: usize,
    pub viewpoint_origin: Vector3<f32>,
    pub viewpoint_quat: Quaternion<f32>,
    pub data: PcdData,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pcd {
    pub header: PcdHeader,
    pub finite: bool,
    pub data: Vec<u8>,
}

impl Pcd {
    pub fn read<R: BufRead>(mut reader: R) -> Result<Self, Box<dyn Error>> {
        let header = PcdHeader::read(&mut reader)?;
        let mut data = Vec::new();
        let finite = header.data.read(reader, &header.fields, &mut data)?;
        Ok(Pcd {
            header,
            finite,
            data,
        })
    }

    pub fn write<W: Write>(&self, mut writer: W) -> Result<(), Box<dyn Error>> {
        self.header.write(&mut writer)?;
        self.header.data.write(&self.data, &self.header, writer)?;
        Ok(())
    }
}

#[inline]
pub fn read_pcd<P, R: BufRead>(reader: R) -> Result<(PointCloud<P>, Viewpoint), Box<dyn Error>>
where
    P: Point + PointFields,
    P::Data: ComplexField,
{
    let pcd = Pcd::read(reader)?;
    pcd.to_point_cloud()
}
