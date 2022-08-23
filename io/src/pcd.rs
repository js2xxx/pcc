mod convert;
mod read;
mod write;

use std::{
    any::TypeId,
    error::Error,
    io::{BufRead, Write},
};

use nalgebra::{ComplexField, Quaternion, Scalar, Vector3};
use pcc_common::{
    point::{Data, DataFields},
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

impl PcdField {
    pub fn from_info<T: PcdFieldData>(field: pcc_common::point::FieldInfo) -> Self {
        PcdField {
            name: field.name.to_string(),
            ty: T::FIELD_TYPE,
            count: field.len,
        }
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

pub trait PcdFieldData: Scalar {
    const FIELD_TYPE: PcdFieldType;
}

macro_rules! impl_pcd_field_data {
    ($($type:ty => $value:ident),*) => {
        $(
            impl PcdFieldData for $type {
                const FIELD_TYPE: PcdFieldType = PcdFieldType:: $value;
            }
        )*
    };
}
impl_pcd_field_data!(
    u8 => U8, i8 => I8, u16 => U16, i16 => I16, u32 => U32, i32 => I32, f32 => F32,
    u64 => U64, i64 => I64, f64 => F64, u128 => U128, i128 => I128
);

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
pub fn read_pcd<P, R>(reader: R) -> Result<(PointCloud<P>, Viewpoint), Box<dyn Error>>
where
    R: BufRead,
    P: Data + DataFields,
    P::Data: ComplexField,
{
    let pcd = Pcd::read(reader)?;
    pcd.to_point_cloud()
}

#[inline]
pub fn write_pcd<P, W>(
    point_cloud: &PointCloud<P>,
    viewpoint: &Viewpoint,
    data_type: PcdData,
    writer: W,
) -> Result<(), Box<dyn Error>>
where
    W: Write,
    P: Data + DataFields,
    P::Data: PcdFieldData,
{
    Pcd::from_point_cloud(point_cloud, viewpoint, data_type).write(writer)
}

#[cfg(test)]
mod tests {
    use std::io::{BufReader, Seek, SeekFrom};

    use nalgebra::Vector4;
    use pcc_common::{
        point::{Normal, Point, Point3LN, PointLabel},
        point_cloud::PointCloud,
    };

    use super::PcdData;
    use crate::pcd::Pcd;

    #[test]
    fn test_io_pcd() {
        let pc = PointCloud::from_vec(
            vec![
                Point3LN::default()
                    .with_coords(nalgebra::Point3::new(2.0, 3.0, 4.0).to_homogeneous())
                    .with_normal(Vector4::new(-1., -2., -3., 0.))
                    .with_curvature(0.5)
                    .with_label(0xABCD);
                4
            ],
            2,
        );

        let mut file = tempfile::tempfile().expect("Failed to open test file");

        let pcd = Pcd::from_point_cloud(&pc, &Default::default(), PcdData::BinaryCompressed);

        pcd.write(&mut file).expect("Failed to write test file");

        file.seek(SeekFrom::Start(0))
            .expect("Failed to seek to start");

        let pcd2 = Pcd::read(BufReader::new(file)).expect("Failed to read test file");

        assert_eq!(pcd, pcd2);

        let (pc2, _) = pcd2
            .to_point_cloud()
            .expect("Failed to convert point cloud");

        assert_eq!(pc, pc2);
    }
}
