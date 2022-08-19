pub struct FieldInfo {
    pub name: &'static str,
    pub offset: usize,
    pub len: usize,
    pub space_len: usize,
}

impl FieldInfo {
    #[inline]
    pub const fn single(name: &'static str, offset: usize) -> Self {
        FieldInfo {
            name,
            offset,
            len: 1,
            space_len: 1,
        }
    }

    #[inline]
    pub fn dim3(name: &'static str, offset: usize) -> Self {
        assert_eq!(offset % 4, 0);
        FieldInfo {
            name,
            offset,
            len: 3,
            space_len: 4,
        }
    }
}

pub trait PointFields {
    type Iter: Iterator<Item = FieldInfo>;

    fn fields() -> Self::Iter;
}
