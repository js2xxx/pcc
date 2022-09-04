use nalgebra::{convert, DVector, RealField, Scalar, Vector4};
use num::ToPrimitive;
use pcc_common::{
    feature::Feature,
    point::{Normal, Point},
    point_cloud::PointCloud,
};

use crate::{pfh::PfhPair, HIST_MAX};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct VfhEstimation<T: Scalar> {
    pub subdivision: [usize; 4],
    pub viewpoint: Vector4<T>,
    pub subd_vp: usize,
    pub normal: Option<Vector4<T>>,
    pub centroid: Option<Vector4<T>>,
    pub has_size: bool,
}

impl<T: Scalar> VfhEstimation<T> {
    #[inline]
    pub fn new(
        subdivision: [usize; 4],
        viewpoint: Vector4<T>,
        subd_vp: usize,
        normal: Option<Vector4<T>>,
        centroid: Option<Vector4<T>>,
        has_size: bool,
    ) -> Self {
        VfhEstimation {
            subdivision,
            viewpoint,
            subd_vp,
            normal,
            centroid,
            has_size,
        }
    }
}

impl<T: RealField + ToPrimitive> VfhEstimation<T> {
    fn point_spfh<P, N>(
        &self,
        points: &[P],
        normals: &[N],
        cp: &Vector4<T>,
        cn: &Vector4<T>,
    ) -> [DVector<T>; 4]
    where
        P: Point<Data = T>,
        N: Normal<Data = T>,
    {
        let num = self.subdivision.map(|sub| convert::<_, T>(sub as f64));
        let mut hist = self.subdivision.map(DVector::zeros);
        let max_distance = points.iter().fold(T::zero(), |acc, point| {
            acc.max((point.coords() - cp).norm())
        });
        let inc = convert::<_, T>(HIST_MAX) / convert::<_, T>((points.len() - 1) as f64);

        for (point, normal) in points.iter().zip(normals) {
            let pair = match PfhPair::try_new(
                &[cp.xyz(), cn.xyz()],
                &[point.coords().xyz(), normal.normal().xyz()],
            ) {
                Some(pair) => pair,
                None => continue,
            };

            let data = [
                ((pair.theta.clone() + T::pi()) / T::two_pi() * num[0].clone()),
                ((pair.alpha.clone() + T::one()) / convert(2.) * num[1].clone()),
                ((pair.phi.clone() + T::one()) / convert(2.) * num[2].clone()),
            ];
            for ((data, num), hist) in data.into_iter().zip(num.clone()).zip(hist.iter_mut()) {
                let index = { data.clamp(T::zero(), num).floor() }.to_usize().unwrap();
                hist[index] += inc.clone();
            }

            if self.has_size {
                let data = pair.distance / max_distance.clone() * num[3].clone();
                let index = { data.clamp(T::zero(), num[3].clone()).floor() }
                    .to_usize()
                    .unwrap();
                hist[3][index] += inc.clone();
            }
        }
        hist
    }

    fn normal_spfh<N>(&self, normals: &[N], cn: &Vector4<T>) -> DVector<T>
    where
        N: Normal<Data = T>,
    {
        let num: T = convert(self.subd_vp as f64);
        let mut hist = DVector::zeros(self.subd_vp);
        let inc = convert::<_, T>(HIST_MAX) / convert((normals.len() - 1) as f64);

        for normal in normals {
            let data = (normal.normal().dot(cn) + T::one()) / convert(2.) * num.clone();
            let index = { data.clamp(T::zero(), num.clone()).floor() }
                .to_usize()
                .unwrap();
            hist[index] += inc.clone();
        }

        hist
    }
}

impl<'a, 'b, T, I, N> Feature<(&'a PointCloud<I>, &'b PointCloud<N>), DVector<T>, (), ()>
    for VfhEstimation<T>
where
    T: RealField + ToPrimitive,
    I: Point<Data = T> + 'a,
    N: Normal<Data = T> + 'b,
{
    fn compute(
        &self,
        (input, normals): (&'a PointCloud<I>, &'b PointCloud<N>),
        _: (),
        _: (),
    ) -> DVector<T> {
        let cp = { self.centroid.clone() }.unwrap_or_else(|| input.centroid_coords().0.unwrap());
        let cn = { self.normal.clone() }.unwrap_or_else(|| {
            let (acc, num) = if normals.is_bounded() {
                normals.iter().fold((Vector4::zeros(), 0), |(acc, num), v| {
                    (acc + v.normal(), num + 1)
                })
            } else {
                normals.iter().fold((Vector4::zeros(), 0), |(acc, num), v| {
                    if v.is_finite() {
                        (acc + v.normal(), num + 1)
                    } else {
                        (acc, num)
                    }
                })
            };

            acc / <T>::from_usize(num).unwrap()
        });

        let vd = (&self.viewpoint - &cp).normalize();

        let [h0, h1, h2, h3] = self.point_spfh(input, normals, &cp, &cn);
        let hn = self.normal_spfh(normals, &vd);

        let mut ret = Vec::from(h0.data);
        ret.append(&mut h1.data.into());
        ret.append(&mut h2.data.into());
        ret.append(&mut h3.data.into());
        ret.append(&mut hn.data.into());
        ret.into()
    }
}
