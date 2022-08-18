use std::{error::Error, io::BufRead};

use nalgebra::{Quaternion, Vector3};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PcdField {
    pub name: String,
    pub size: usize,
    pub ty: PcdFieldType,
    pub count: usize,
}

impl Default for PcdField {
    fn default() -> Self {
        Self {
            name: Default::default(),
            size: 4,
            ty: PcdFieldType::Float,
            count: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PcdFieldType {
    SignedInt,
    UnsignedInt,
    Float,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PcdDataType {
    Ascii,
    Binary,
    BinaryCompressed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcdHeader {
    pub fields: Vec<PcdField>,
    pub width: usize,
    pub height: usize,
    pub viewpoint_origin: Vector3<f32>,
    pub viewpoint_quat: Quaternion<f32>,
    pub data_type: PcdDataType,
}

impl PcdHeader {
    pub fn parse<R: BufRead>(mut reader: R) -> Result<Self, Box<dyn Error>> {
        let mut string = String::new();

        let mut fields = Vec::new();
        let mut width = None;
        let mut height = None;
        let mut viewpoint_origin = Vector3::zeros();
        let mut viewpoint_quat = Quaternion::identity();
        let data_type;

        loop {
            reader.read_line(&mut string)?;

            if string.starts_with('#') {
                continue;
            }

            let (ty, data) = string
                .split_once(' ')
                .ok_or(format!("Non-header data: {:?}", string))?;

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
                        fields[index].size = size.parse()?;
                    }
                }
                "TYPE" => {
                    for (index, ty) in data.split_whitespace().enumerate() {
                        fields[index].ty = match ty {
                            "I" => PcdFieldType::SignedInt,
                            "U" => PcdFieldType::UnsignedInt,
                            "F" => PcdFieldType::Float,
                            _ => return Err(format!("Unknown type {:?}", ty).into()),
                        }
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
                        "ascii" => PcdDataType::Ascii,
                        "binary" => PcdDataType::Binary,
                        "binary_compressed" => PcdDataType::BinaryCompressed,
                        _ => return Err(format!("Unknown data type: {:?}", data).into()),
                    };
                    break;
                }
                _ => {}
            }
        }

        Ok(PcdHeader {
            fields,
            width: width.unwrap(),
            height: height.unwrap(),
            viewpoint_origin,
            viewpoint_quat,
            data_type,
        })
    }
}
