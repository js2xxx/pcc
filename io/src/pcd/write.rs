use std::{error::Error, io::Write};

use super::{PcdData, PcdFieldType, PcdHeader};

impl PcdHeader {
    pub fn write<W>(&self, mut writer: W) -> Result<(), Box<dyn Error>>
    where
        W: Write,
    {
        writeln!(writer, "VERSION .7")?;

        write!(writer, "FIELDS")?;
        for field in &self.fields {
            write!(writer, " {}", field.name)?;
        }
        writeln!(writer)?;

        write!(writer, "SIZE")?;
        for field in &self.fields {
            write!(writer, " {}", field.ty.size())?;
        }
        writeln!(writer)?;

        write!(writer, "TYPE")?;
        for field in &self.fields {
            write!(writer, " {}", field.ty.type_str())?;
        }
        writeln!(writer)?;

        write!(writer, "COUNT")?;
        for field in &self.fields {
            write!(writer, " {}", field.count)?;
        }
        writeln!(writer)?;

        writeln!(writer, "WIDTH {}", self.width)?;
        writeln!(writer, "HEIGHT {}", self.height)?;

        let vec = self.viewpoint_quat.as_vector();
        writeln!(
            writer,
            "VIEWPOINT {} {} {} {} {} {} {}",
            self.viewpoint_origin.x,
            self.viewpoint_origin.y,
            self.viewpoint_origin.z,
            vec.x,
            vec.y,
            vec.z,
            vec.w,
        )?;

        writeln!(writer, "POINTS {}", self.width * self.height)?;
        writeln!(writer, "DATA {}", self.data.type_str())?;

        Ok(())
    }
}

impl PcdData {
    pub fn write<W>(
        &self,
        data: &[u8],
        header: &PcdHeader,
        mut writer: W,
    ) -> Result<(), Box<dyn Error>>
    where
        W: Write,
    {
        match self {
            PcdData::Ascii => write_text(data, header, writer),
            PcdData::Binary => writer.write_all(data).map_err(Into::into),
            PcdData::BinaryCompressed => write_bytes_compressed(data, header, writer),
        }
    }
}

fn write_text<W>(data: &[u8], header: &PcdHeader, mut writer: W) -> Result<(), Box<dyn Error>>
where
    W: Write,
{
    for record in data.chunks(header.rec_size) {
        let mut offset = 0;
        for (fi, field_info) in header.fields.iter().enumerate() {
            let size = field_info.ty.size();
            let field = &record[offset..][..(field_info.count * size)];

            macro_rules! write_field {
                ($type:ty) => {
                    for index in 0..field_info.count {
                        write!(
                            writer,
                            "{}",
                            <$type>::from_ne_bytes(field[(index * size)..][..size].try_into()?)
                        )?;
                        if fi < header.fields.len() - 1 || index < field_info.count - 1 {
                            write!(writer, " ")?
                        }
                    }
                };
            }

            use PcdFieldType::*;
            match field_info.ty {
                U8 => write_field!(u8),
                I8 => write_field!(i8),
                U16 => write_field!(u16),
                I16 => write_field!(i16),
                U32 => write_field!(u32),
                I32 => write_field!(i32),
                F32 => write_field!(f32),
                U64 => write_field!(u64),
                I64 => write_field!(i64),
                F64 => write_field!(f64),
                U128 => write_field!(u128),
                I128 => write_field!(i128),
            };

            offset += field_info.count * size;
        }
        writeln!(writer)?;
    }

    Ok(())
}

fn write_bytes_compressed<W>(
    data: &[u8],
    header: &PcdHeader,
    mut writer: W,
) -> Result<(), Box<dyn Error>>
where
    W: Write,
{
    let record_size = header.rec_size;
    let record_num = data.len() / record_size;

    let mut temp = Vec::with_capacity(data.len());
    {
        let mut offset = 0;
        for field in &header.fields {
            let field_size = field.ty.size() * field.count;
            for record_index in 0..record_num {
                temp.extend_from_slice(
                    &data[(record_size * record_index + offset)..][..field_size],
                );
            }
            offset += field_size;
        }
    }

    let out = crate::lzf::compress(&temp).map_err(|_| "Compression error")?;
    writer.write_all(&(out.len() as u32).to_ne_bytes())?;
    writer.write_all(&(data.len() as u32).to_ne_bytes())?;
    writer.write_all(&out)?;
    Ok(())
}
