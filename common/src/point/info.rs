#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FieldInfo {
    pub name: &'static str,
    // pub ty: TypeId,
    pub offset: usize,
    pub len: usize,
    pub space_len: usize,
}

impl FieldInfo {
    #[inline]
    pub const fn single<T: 'static>(name: &'static str, offset: usize) -> Self {
        FieldInfo {
            name,
            // ty: TypeId::of::<T>(),
            offset,
            len: 1,
            space_len: 1,
        }
    }

    #[inline]
    pub fn dim3<T: 'static>(name: &'static str, offset: usize) -> Self {
        assert_eq!(offset % 4, 0);
        FieldInfo {
            name,
            // ty: TypeId::of::<T>(),
            offset,
            len: 3,
            space_len: 4,
        }
    }
}

pub trait PointFields {
    type Iter: Iterator<Item = FieldInfo> + Clone;

    fn fields() -> Self::Iter;
}
