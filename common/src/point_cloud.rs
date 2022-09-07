mod reference;
mod transforms;

use std::{
    borrow::Cow,
    fmt::Debug,
    ops::{Deref, Index, IndexMut},
};

use nalgebra::{ComplexField, RealField, Vector4};

pub use self::reference::{AsPointCloud, PointCloudRef};
use self::transforms::Transform;
use crate::point::{Data, Point};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointCloud<P> {
    storage: Vec<P>,
    width: usize,
    bounded: bool,
}

impl<P> PointCloud<P> {
    #[inline]
    pub fn width(&self) -> usize {
        assert_eq!(self.storage.len() % self.width, 0);
        self.width
    }

    #[inline]
    pub fn index(&self, index: usize) -> [usize; 2] {
        [index % self.width, index / self.width]
    }

    #[inline]
    pub fn height(&self) -> usize {
        assert_eq!(self.storage.len() % self.width, 0);
        self.storage.len() / self.width
    }

    #[inline]
    pub fn is_bounded(&self) -> bool {
        self.bounded
    }

    #[inline]
    pub fn into_vec(self) -> Vec<P> {
        self.storage
    }

    /// # Safety
    ///
    /// The width and boundedness of the point cloud must be valid.
    #[inline]
    pub unsafe fn storage(&mut self) -> &mut Vec<P> {
        &mut self.storage
    }

    #[inline]
    pub fn select<'a>(&'a self, indices: Cow<'a, [usize]>) -> PointCloudRef<'a, P> {
        PointCloudRef::new(self, Some(indices))
    }
}

impl<P: Clone> PointCloud<P> {
    pub fn try_create_sub(&self, indices: &[usize], width: usize) -> Option<Self> {
        (width > 0 && indices.len() % width == 0).then(|| PointCloud {
            storage: { indices.iter() }
                .map(|&index| self.storage[index].clone())
                .collect(),
            width,
            bounded: self.bounded,
        })
    }

    #[inline]
    pub fn create_sub(&self, indices: &[usize], width: usize) -> Self {
        self.try_create_sub(indices, width)
            .expect("The length of the vector must be divisible by width")
    }
}

impl<P> Deref for PointCloud<P> {
    type Target = [P];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}

impl<P> Index<usize> for PointCloud<P> {
    type Output = P;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.storage[index]
    }
}

impl<P> IndexMut<usize> for PointCloud<P> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.storage[index]
    }
}

impl<P> Index<(usize, usize)> for PointCloud<P> {
    type Output = P;

    #[inline]
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        &self.storage[y * self.width + x]
    }
}

impl<P> IndexMut<(usize, usize)> for PointCloud<P> {
    #[inline]
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        &mut self.storage[y * self.width + x]
    }
}

impl<P> PointCloud<P> {
    #[inline]
    pub fn new() -> Self {
        PointCloud {
            storage: Vec::new(),
            width: 0,
            bounded: true,
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that the points in the cloud are all `bounded`
    /// and the length of `storage` is divisible by `width`.
    #[inline]
    pub unsafe fn from_raw_parts(storage: Vec<P>, width: usize, bounded: bool) -> Self {
        PointCloud {
            storage,
            width,
            bounded,
        }
    }
}

impl<P: Clone> PointCloud<P> {
    #[inline]
    pub fn transpose(&self) -> Self {
        let mut other = Self::new();
        self.transpose_into(&mut other);
        other
    }

    pub fn transpose_into(&self, other: &mut Self) {
        other.storage.clear();
        other.storage.reserve(self.storage.len());

        let width = self.width;
        unsafe {
            let space = other.storage.spare_capacity_mut();

            for (index, obj) in self.storage.iter().enumerate() {
                let [x, y] = self.index(index);
                space[x * width + y].write(obj.clone());
            }

            other.storage.set_len(other.storage.capacity());
        }

        other.width = self.height();
        other.bounded = self.bounded;
    }
}

impl<P: Data> PointCloud<P> {
    pub fn try_from_vec(storage: Vec<P>, width: usize) -> Result<Self, Vec<P>> {
        if width > 0 && storage.len() % width == 0 {
            let bounded = storage.iter().all(|p| p.is_finite());
            Ok(PointCloud {
                storage,
                width,
                bounded,
            })
        } else {
            Err(storage)
        }
    }

    pub fn reinterpret(&mut self, width: usize) {
        assert!(width > 0);
        assert_eq!(self.storage.len() % width, 0);
        self.width = width;
        self.bounded = self.storage.iter().all(|p| p.is_finite());
    }

    #[inline]
    pub fn from_vec(storage: Vec<P>, width: usize) -> Self {
        PointCloud::try_from_vec(storage, width)
            .expect("The length of the vector must be divisible by width")
    }
}

impl<P> Default for PointCloud<P> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Point> PointCloud<P> where P::Data: RealField {}

impl<P: Point> PointCloud<P>
where
    <P as Data>::Data: ComplexField,
{
    pub fn transform<Z: Transform<P::Data>>(&self, z: &Z, out: &mut Self) {
        out.storage
            .resize_with(self.storage.len(), Default::default);

        out.width = self.width;
        out.bounded = self.bounded;

        if self.bounded {
            for (from, to) in self.storage.iter().zip(out.storage.iter_mut()) {
                z.se3(from.coords(), to.coords_mut())
            }
        } else {
            for (from, to) in self.storage.iter().zip(out.storage.iter_mut()) {
                if !from.is_finite() {
                    continue;
                }
                z.se3(from.coords(), to.coords_mut())
            }
        }
    }

    pub fn map<F, R>(&self, f: F) -> PointCloud<R>
    where
        F: FnMut(&P) -> R,
        R: Data,
    {
        let iter = self.storage.iter().map(f);
        PointCloud::from_vec(iter.collect(), self.width)
    }
}

impl<P: Point> PointCloud<P>
where
    <P as Data>::Data: ComplexField,
{
    #[inline]
    pub fn demean(&self, centroid: &Vector4<P::Data>, out: &mut Self) {
        out.clone_from(self);
        out.demean_mut(centroid);
    }

    pub fn demean_mut(&mut self, centroid: &Vector4<P::Data>) {
        for point in &mut self.storage {
            point.coords_mut().x -= centroid.x.clone();
            point.coords_mut().y -= centroid.y.clone();
            point.coords_mut().z -= centroid.z.clone();
        }
    }
}
