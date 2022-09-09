use nalgebra::{convert, Matrix3, RealField, Vector3};
use pcc_common::{
    feature::Feature,
    point::{Normal, PointIntensity},
    point_cloud::PointCloud,
    search::{Search, SearchType},
};
use rayon::prelude::*;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IntensityGradient;

impl IntensityGradient {
    fn point<T, P>(
        &self,
        input: &PointCloud<P>,
        result: &[(usize, T)],
        pivot: Vector3<T>,
        normal: Vector3<T>,
        mean_intensity: T,
    ) -> Option<Vector3<T>>
    where
        T: RealField,
        P: PointIntensity<Data = T>,
    {
        if result.len() < 3 {
            return None;
        }

        let iter = result.iter().filter_map(|&(index, _)| {
            let point = &input[index];
            if !point.is_finite() || !point.intensity().is_finite() {
                return None;
            }

            let diff = point.na_point().coords - &pivot;
            let diff_intensity = point.intensity() - mean_intensity.clone();

            Some((diff, diff_intensity))
        });

        let (mat, vec) = iter.fold(
            (Matrix3::zeros(), Vector3::zeros()),
            |(mut mat, mut vec), (diff, diff_intensity)| {
                mat.syger(T::one(), &diff, &diff, T::one());
                vec.axpy(diff_intensity, &diff, T::one());
                (mat, vec)
            },
        );

        let eigen = mat.symmetric_eigen();
        let mut b = eigen.eigenvectors.tr_mul(&vec);
        for (ev, b) in eigen.eigenvalues.iter().zip(&mut b) {
            if ev.is_zero() {
                *b = T::zero();
            } else {
                *b /= ev.clone()
            }
        }

        let x = eigen.eigenvectors * b;

        Some((Matrix3::identity() - &normal * normal.transpose()) * x)
    }
}

impl<'a, T, P, N, S>
    Feature<(&'a PointCloud<P>, &'a PointCloud<N>), PointCloud<Vector3<T>>, S, SearchType<T>>
    for IntensityGradient
where
    T: RealField + Default,
    P: Sync + PointIntensity<Data = T>,
    N: Sync + Normal<Data = T>,
    S: Sync + Search<'a, P>,
{
    fn compute(
        &self,
        (input, normals): (&'a PointCloud<P>, &'a PointCloud<N>),
        search: S,
        ty: SearchType<T>,
    ) -> PointCloud<Vector3<T>> {
        fn collect<T: Send + Sync>(
            iter: impl ParallelIterator<Item = (bool, Vector3<T>)>,
            init: bool,
        ) -> (Vec<Vector3<T>>, bool) {
            let fold = iter.fold(
                || (Vec::new(), init),
                |(mut storage, bounded), (b2, gradient)| {
                    storage.push(gradient);
                    (storage, bounded & b2)
                },
            );
            fold.reduce(
                || (Vec::new(), true),
                |(mut sa, ba), (mut sb, bb)| {
                    sa.append(&mut sb);
                    (sa, ba & bb)
                },
            )
        }

        let zip = input.par_iter().zip(normals.par_iter());

        let (storage, bounded) = if input.is_bounded() {
            let iter = zip.map(|(point, normal)| {
                let mut result = Vec::new();
                search.search(point.coords(), ty.clone(), &mut result);
                if result.is_empty() {
                    return (false, Vector3::zeros());
                }
                let (point, intensity) = result.iter().fold(
                    (Vector3::zeros(), T::zero()),
                    |(point, intensity), &(index, _)| {
                        (
                            point + search.input()[index].coords().xyz(),
                            intensity + search.input()[index].intensity(),
                        )
                    },
                );
                let num: T = convert(result.len() as f64);
                let gradient = self.point(
                    input,
                    &result,
                    point / num.clone(),
                    normal.normal().xyz(),
                    intensity / num,
                );
                (true, gradient.unwrap_or_default())
            });
            collect(iter, true)
        } else {
            let iter = zip.map(|(point, normal)| {
                if !point.is_finite() {
                    return (false, Vector3::zeros());
                }
                let mut result = Vec::new();
                search.search(point.coords(), ty.clone(), &mut result);
                if result.is_empty() {
                    return (false, Vector3::zeros());
                }
                let (point, intensity) = result.iter().fold(
                    (Vector3::zeros(), T::zero()),
                    |(point, intensity), &(index, _)| {
                        (
                            point + search.input()[index].coords().xyz(),
                            intensity + search.input()[index].intensity(),
                        )
                    },
                );
                let num: T = convert(result.len() as f64);
                let gradient = self.point(
                    input,
                    &result,
                    point / num.clone(),
                    normal.normal().xyz(),
                    intensity / num,
                );
                (true, gradient.unwrap_or_default())
            });
            collect(iter, true)
        };

        unsafe { PointCloud::from_raw_parts(storage, input.width(), bounded) }
    }
}
