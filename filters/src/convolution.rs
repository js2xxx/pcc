mod gauss;

use std::fmt::Debug;

use nalgebra::{ComplexField, DVector, Scalar, Vector4};
use pcc_common::{
    point::Point,
    point_cloud::PointCloud,
    search::{SearchType, Searcher},
};
use rayon::{iter::ParallelIterator, prelude::IntoParallelRefIterator};

pub use self::gauss::{Gauss, GaussRgba};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BorderOptions {
    Default,
    Mirrored,
    Repeated,
}

/// This struct only processes organized point clouds (2-D indices-wise)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fixed2<T: Scalar> {
    pub kernel: DVector<T>,
    pub border_options: BorderOptions,
}

impl<T: Scalar> Fixed2<T> {
    pub fn new(kernel: DVector<T>, border_options: BorderOptions) -> Self {
        Fixed2 {
            kernel,
            border_options,
        }
    }
}

impl<T: ComplexField> Fixed2<T> {
    fn convolve_one<P: Point<Data = T>>(&self, points: &[P], kernel_len: usize) -> Option<P> {
        let (sum, weight) = (0..kernel_len).zip(points.iter().rev()).fold(
            (Vector4::zeros(), T::zero()),
            |(sum, w), (k, point)| {
                (
                    sum + point.coords().clone() * self.kernel[k].clone(),
                    w + self.kernel[k].clone(),
                )
            },
        );

        (weight != T::zero()).then(|| P::default().with_coords(sum / weight))
    }

    fn convolve_default<P: Point<Data = T> + Clone>(
        &self,
        points: &[P],
        seg_size: usize,
        storage: &mut Vec<P>,
    ) {
        storage.clear();
        storage.reserve(points.len());

        let kernel_len = self.kernel.len();
        let start_gap = kernel_len / 2;

        for seg in (0..points.len()).step_by(seg_size) {
            storage.resize_with(seg + start_gap, Default::default);
            for index in seg..=(seg + seg_size - kernel_len) {
                let point = self.convolve_one(&points[index..][..kernel_len], kernel_len);
                storage.push(point.unwrap());
            }
            storage.resize_with(seg + seg_size, Default::default)
        }
    }

    fn convolve_mirrored<P: Point<Data = T> + Clone>(
        &self,
        points: &[P],
        seg_size: usize,
        storage: &mut Vec<P>,
    ) {
        storage.clear();
        storage.reserve(points.len());

        let kernel_len = self.kernel.len();
        let start_gap = kernel_len / 2;
        let end_gap = kernel_len - start_gap - 1;
        let last = seg_size - end_gap - 1;

        for seg in (0..points.len()).step_by(seg_size) {
            storage.resize_with(seg + start_gap, Default::default);
            for index in seg..=(seg + seg_size - kernel_len) {
                let point = self.convolve_one(&points[index..][..kernel_len], kernel_len);
                storage.push(point.unwrap());
            }
            for index in 0..end_gap {
                let point = storage[seg + last - index].clone();
                storage.push(point);
            }
            for index in 0..start_gap {
                let point = storage[seg + start_gap + index].clone();
                storage[seg + start_gap - index - 1] = point;
            }
        }
    }

    fn convolve_repeated<P: Point<Data = T> + Clone>(
        &self,
        points: &[P],
        seg_size: usize,
        storage: &mut Vec<P>,
    ) {
        storage.clear();
        storage.reserve(points.len());

        let kernel_len = self.kernel.len();
        let start_gap = kernel_len / 2;
        let last = seg_size - kernel_len + start_gap;

        for seg in (0..points.len()).step_by(seg_size) {
            storage.resize_with(seg + start_gap, Default::default);
            for index in seg..=(seg + seg_size - kernel_len) {
                let point = self.convolve_one(&points[index..][..kernel_len], kernel_len);
                storage.push(point.unwrap());
            }

            let point = storage[seg + last].clone();
            storage.resize_with(seg + seg_size, || point.clone());

            let point = storage[seg + start_gap].clone();
            for index in 0..start_gap {
                storage[seg + index] = point.clone();
            }
        }
    }

    pub fn convolve_rows_into<P: Point<Data = T> + Clone + Debug>(
        &self,
        input: &PointCloud<P>,
        output: &mut PointCloud<P>,
    ) {
        let width = input.width();
        unsafe {
            match self.border_options {
                BorderOptions::Default => self.convolve_default(input, width, output.storage()),
                BorderOptions::Mirrored => self.convolve_mirrored(input, width, output.storage()),
                BorderOptions::Repeated => self.convolve_repeated(input, width, output.storage()),
            }

            output.reinterpret(width)
        }
    }

    pub fn convolve_rows<P: Point<Data = T> + Clone + Debug>(
        &self,
        input: &PointCloud<P>,
    ) -> PointCloud<P> {
        let mut output = PointCloud::new();
        self.convolve_rows_into(input, &mut output);

        output
    }

    pub fn convolve_columns_into<P: Point<Data = T> + Clone + Debug>(
        &self,
        input: &PointCloud<P>,
        output: &mut PointCloud<P>,
    ) {
        input.transpose_into(output);

        let temp = self.convolve_rows(output);
        temp.transpose_into(output);
    }

    pub fn convolve_columns<P: Point<Data = T> + Clone + Debug>(
        &self,
        input: &PointCloud<P>,
    ) -> PointCloud<P> {
        let mut output = PointCloud::new();
        self.convolve_columns_into(input, &mut output);

        output
    }

    pub fn convolve<P: Point<Data = T> + Clone + Debug>(
        &self,
        input: &PointCloud<P>,
    ) -> PointCloud<P> {
        let mut transposed = PointCloud::new();

        let mut temp = self.convolve_rows(input);

        temp.transpose_into(&mut transposed);
        self.convolve_rows_into(&transposed, &mut temp);
        temp.transpose_into(&mut transposed);

        transposed
    }

    pub fn convolve_into<P: Point<Data = T> + Clone + Debug>(
        &self,
        input: &PointCloud<P>,
        output: &mut PointCloud<P>,
    ) {
        let mut transposed = PointCloud::new();

        self.convolve_rows_into(input, output);

        output.transpose_into(&mut transposed);
        self.convolve_rows_into(&transposed, output);
        output.transpose_into(&mut transposed);

        *output = transposed;
    }
}

pub trait DynamicKernel<'a, P: Point + 'a> {
    fn convolve<Iter>(&self, data: Iter) -> P
    where
        Iter: IntoIterator<Item = (&'a P, P::Data)>;
}

/// This struct proceesses point clouds 3-D coordinates-wise, and provides
/// neighbors within a specific radius for each point to the kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dynamic<T: Scalar, K, S> {
    pub kernel: K,
    pub searcher: S,
    pub radius: T,
}

impl<T: Scalar, K, S> Dynamic<T, K, S> {
    pub fn new(kernel: K, searcher: S, radius: T) -> Self {
        Dynamic {
            kernel,
            searcher,
            radius,
        }
    }
}

impl<'a, T: ComplexField, K, S> Dynamic<T, K, S> {
    pub fn convolve_par<P>(&self) -> PointCloud<P>
    where
        P: Sync + Send + Point<Data = T> + 'a,
        K: Sync + DynamicKernel<'a, P>,
        S: Sync + Searcher<'a, P>,
    {
        let input = self.searcher.point_cloud();

        let output = { input.par_iter() }
            .map(|point| {
                let mut result = Vec::new();

                self.searcher.search(
                    point.coords(),
                    SearchType::Radius(self.radius.clone()),
                    &mut result,
                );

                self.kernel.convolve(
                    { result.into_iter() }.map(|(index, distance)| (&input[index], distance)),
                )
            })
            .collect::<Vec<_>>();

        PointCloud::from_vec(output, input.width())
    }

    pub fn convolve<P>(&self) -> PointCloud<P>
    where
        P: Point<Data = T> + 'a,
        K: DynamicKernel<'a, P>,
        S: Searcher<'a, P>,
    {
        let input = self.searcher.point_cloud();

        let output = { input.iter() }
            .map(|point| {
                let mut result = Vec::new();

                self.searcher.search(
                    point.coords(),
                    SearchType::Radius(self.radius.clone()),
                    &mut result,
                );

                self.kernel.convolve(
                    { result.into_iter() }.map(|(index, distance)| (&input[index], distance)),
                )
            })
            .collect::<Vec<_>>();

        PointCloud::from_vec(output, input.width())
    }
}
