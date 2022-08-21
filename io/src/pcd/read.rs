use std::{error::Error, io::BufRead};

use nalgebra::{Quaternion, Vector3};

use super::{PcdData, PcdField, PcdFieldType, PcdHeader};

impl PcdField {
    fn read_text<'a, I: Iterator<Item = &'a str>, E: Extend<u8>>(
        &self,
        mut data: I,
        output: &mut E,
    ) -> Result<bool, Box<dyn Error>> {
        let mut finite = true;
        for _ in 0..self.count {
            macro_rules! read_field {
                ($var:expr, {$($value:pat => $out:ty $(|$temp:ident| $temp_body:block)?),*}) => {
                    match $var {
                        $($value => {
                            let data = data.next().ok_or("Not enough fields")?.parse::<$out>()?;
                            $(
                                let $temp = data;
                                $temp_body
                                let data = $temp;
                            )?
                            output.extend(data.to_ne_bytes())
                        })*
                    }
                };
            }
            use PcdFieldType::*;
            read_field! (self.ty, {
                I8 => i8, U8 => u8,
                I16 => i16, U16 => u16,
                I32 => i32, U32 => u32,
                F32 => f32 |data| { finite &= data.is_finite() },
                I64 => i64, U64 => u64,
                F64 => f64 |data| { finite &= data.is_finite() },
                I128 => i128, U128 => u128
            });
        }
        Ok(finite)
    }

    fn check_binary(&self, data: &mut &[u8]) -> bool {
        let mut finite = true;
        for _ in 0..self.count {
            macro_rules! read_field {
                ($var:expr, {$($pat:pat => $value:expr, $out:ty $(|$temp:ident| $temp_body:block)?),*}) => {
                    match $var {
                        $($pat => {
                            let size = $value .size();
                            $(
                                let $temp = <$out>::from_ne_bytes((*data)[..size].try_into().unwrap());
                                $temp_body
                            )?
                            *data = &(*data)[size..];
                        })*
                    }
                };
            }
            use PcdFieldType::*;
            read_field! (self.ty, {
                I8   => I8,   i8,   U8   => U8,   u8,
                I16  => I16,  i16,  U16  => U16,  u16,
                I32  => I32,  i32,  U32  => U32,  u32,
                F32  => F32,  f32 |data| { finite &= data.is_finite() },
                I64  => I64,  i64,  U64  => U64,  u64,
                F64  => F64,  f64 |data| { finite &= data.is_finite() },
                I128 => I128, I128, U128 => U128, u128
            });
        }
        finite
    }
}

impl PcdHeader {
    pub fn read<R: BufRead>(mut reader: R) -> Result<Self, Box<dyn Error>> {
        let mut string = String::new();

        let mut fields = Vec::new();
        let mut width = None;
        let mut height = None;
        let mut viewpoint_origin = Vector3::zeros();
        let mut viewpoint_quat = Quaternion::identity();
        let data_type;

        loop {
            string.clear();
            let num = reader.read_line(&mut string)?;
            string.pop();
            if num == 0 {
                return Err("Unexpected EOF".into());
            }

            if string.starts_with('#') {
                continue;
            }

            let (ty, data) = string
                .split_once(' ')
                .ok_or_else(|| format!("Non-header data: {:?}", string))?;

            match ty {
                "VERSION" => {}
                "FIELDS" | "COLUMNS" => {
                    fields.clear();
                    fields.extend(data.split_whitespace().map(|name| PcdField {
                        name: name.to_owned(),
                        ..Default::default()
                    }));
                }
                "SIZE" => {
                    for (index, size) in data.split_whitespace().enumerate() {
                        fields[index].ty = PcdFieldType::default_sized(size.parse()?)?;
                    }
                }
                "TYPE" => {
                    for (index, ty) in data.split_whitespace().enumerate() {
                        fields[index].ty.set_type(ty)?;
                    }
                }
                "COUNT" => {
                    for (index, count) in data.split_whitespace().enumerate() {
                        fields[index].count = count.parse()?;
                    }
                }
                "WIDTH" => width = Some(data.parse()?),
                "HEIGHT" => height = Some(data.parse()?),
                "VIEWPOINT" => {
                    for (field, data) in viewpoint_origin
                        .iter_mut()
                        .chain(viewpoint_quat.coords.iter_mut())
                        .zip(data.split_whitespace())
                    {
                        *field = data.parse()?;
                    }
                }
                "POINTS" => {
                    let points = data.parse()?;
                    match (width, height) {
                        (None, None) => {
                            width = Some(points);
                            height = Some(1);
                        }
                        (Some(width), None) => {
                            if points % width == 0 {
                                height = Some(points / width)
                            } else {
                                return Err("POINTS % WIDTH != 0".into());
                            }
                        }
                        (None, Some(height)) => {
                            if points % height == 0 {
                                width = Some(points / height)
                            } else {
                                return Err("POINTS % HEIGHT != 0".into());
                            }
                        }
                        (Some(width), Some(height)) => {
                            if width * height != points {
                                return Err("POINTS conflicts with WIDTH * HEIGHT".into());
                            }
                        }
                    }
                }
                "DATA" => {
                    data_type = match data {
                        "ascii" => PcdData::Ascii,
                        "binary" => PcdData::Binary,
                        "binary_compressed" => PcdData::BinaryCompressed,
                        _ => return Err(format!("Unknown data type: {:?}", data).into()),
                    };
                    break;
                }
                _ => {}
            }
        }
        let rec_size = { fields.iter() }.fold(0, |acc, field| acc + field.count * field.ty.size());

        Ok(PcdHeader {
            fields,
            rec_size,
            width: width.unwrap(),
            height: height.unwrap(),
            viewpoint_origin,
            viewpoint_quat,
            data: data_type,
        })
    }
}

impl PcdData {
    pub fn read<R: BufRead>(
        &self,
        reader: R,
        fields: &[PcdField],
        output: &mut Vec<u8>,
    ) -> Result<bool, Box<dyn Error>> {
        output.clear();
        match self {
            PcdData::Ascii => read_text(reader, fields, output),
            PcdData::Binary => read_bytes::<_, false>(reader, fields, output),
            PcdData::BinaryCompressed => read_bytes::<_, true>(reader, fields, output),
        }
    }
}

fn read_text<R: BufRead>(
    reader: R,
    fields: &[PcdField],
    output: &mut Vec<u8>,
) -> Result<bool, Box<dyn Error>> {
    let mut finite = true;
    for string in reader.lines().flatten() {
        let mut data = string.split_whitespace();
        for field in fields {
            finite &= field.read_text(&mut data, output)?
        }
    }
    Ok(finite)
}

fn read_bytes<R: BufRead, const COMPRESS: bool>(
    mut reader: R,
    fields: &[PcdField],
    output: &mut Vec<u8>,
) -> Result<bool, Box<dyn Error>> {
    if COMPRESS {
        let mut buf = [0; 4];
        let compressed_size = {
            reader.read_exact(&mut buf)?;
            u32::from_ne_bytes(buf) as usize
        };
        let uncompressed_size = {
            reader.read_exact(&mut buf)?;
            u32::from_ne_bytes(buf) as usize
        };

        output.resize(compressed_size, 0);
        reader.read_exact(output)?;

        let temp = &*crate::lzf::decompress(output, uncompressed_size as usize)
            .map_err(|_| "Decompression error")?;
        let size = uncompressed_size;
        output.clear();
        output.reserve(size);

        let record_size = fields.iter().fold(0, |acc, field| acc + field.ty.size() * field.count);
        let record_num = size / record_size;

        for record_index in 0..record_num {
            let mut offset = 0;
            for field in fields {
                let field_size = field.ty.size() * field.count;
                output.extend_from_slice(&temp[(offset + field_size * record_index)..][..field_size]);
                offset += field_size * record_num;
            }
        }
    } else {
        reader.read_to_end(output)?;
    }

    let mut finite = true;
    let mut data = &**output;
    for field in fields {
        finite &= field.check_binary(&mut data);
    }
    Ok(finite)
}
