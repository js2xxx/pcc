use std::{array, collections::HashSet, ops::Deref};

use nalgebra::{
    base::dimension::Dynamic, convert, Const, DMatrix, DVector, MatrixSliceMut1xX, RealField,
};
use num::ToPrimitive;
use pcc_common::{
    feature::Feature,
    point::{Normal, Point},
    point_cloud::PointCloud,
    search::{Search, SearchType},
};

use crate::pfh::PfhPair;

pub struct FpfhEstimation<N> {
    pub subdivision: [usize; 3],
    pub normals: N,
}

impl<N> FpfhEstimation<N> {
    pub fn new(subdivision: [usize; 3], normals: N) -> Self {
        FpfhEstimation {
            subdivision,
            normals,
        }
    }
}

impl<N> FpfhEstimation<N> {
    fn point_spfh<T, P, N2>(
        &self,
        pivot: usize,
        indices: &[(usize, T)],
        points: &[P],
        normals: &[N2],
        mut hist: [MatrixSliceMut1xX<'_, T, Const<1>, Dynamic>; 3],
    ) where
        T: RealField + ToPrimitive,
        P: Point<Data = T>,
        N2: Normal<Data = T>,
    {
        let num: [T; 3] = array::from_fn(|index| convert(hist[index].ncols() as f64));
        let inc = convert::<_, T>(100.) / convert((indices.len() - 1) as f64);

        for index in indices.iter().map(|&(index, _)| index) {
            if pivot == index {
                continue;
            }
            let pair = {
                let pair = PfhPair::try_new(
                    &[points[pivot].coords().xyz(), normals[pivot].normal().xyz()],
                    &[points[index].coords().xyz(), normals[index].normal().xyz()],
                );
                match pair {
                    Some(pair) => pair,
                    None => continue,
                }
            };
            let data = [
                ((pair.theta.clone() + T::pi()) / T::two_pi() * num[0].clone()),
                ((pair.alpha.clone() + T::one()) / convert(2.) * num[1].clone()),
                ((pair.phi.clone() + T::one()) / convert(2.) * num[2].clone()),
            ];
            for ((data, num), hist) in data.into_iter().zip(num.clone()).zip(hist.iter_mut()) {
                let index = { data.clamp(T::zero(), num).floor() }.to_usize().unwrap();
                hist[index] += inc.clone()
            }
        }
    }
}

impl<V> FpfhEstimation<V> {
    fn compute_spfh<'a, T: RealField, P, S, N>(
        &self,
        input: &PointCloud<P>,
        search: &S,
        ty: SearchType<T>,
    ) -> (Vec<usize>, [DMatrix<T>; 3])
    where
        T: RealField + ToPrimitive,
        P: Point<Data = T> + 'a,
        S: Search<'a, P>,
        V: Deref<Target = [N]>,
        N: Normal<Data = T>,
    {
        let mut result = Vec::new();

        let indices = if search.input() == input {
            (0..input.len()).collect::<HashSet<_>>()
        } else {
            input.iter().fold(HashSet::new(), |mut set, point| {
                search.search(point.coords(), ty.clone(), &mut result);
                set.extend(result.iter().map(|&(index, _)| index));
                set
            })
        };

        let mut ret = vec![0; indices.len()];

        let mut hist: [_; 3] =
            array::from_fn(|index| DMatrix::zeros(indices.len(), self.subdivision[index]));

        for (ii, index) in indices.into_iter().enumerate() {
            search.search(input[index].coords(), ty.clone(), &mut result);
            let [h1, h2, h3] = &mut hist;
            self.point_spfh(
                index,
                &result,
                search.input(),
                &self.normals,
                [h1.row_mut(ii), h2.row_mut(ii), h3.row_mut(ii)],
            );
            ret[index] = ii;
        }

        (ret, hist)
    }

    fn weight_spfh<T: RealField>(
        &self,
        hist: &[DMatrix<T>; 3],
        search_res: &[(usize, T)],
    ) -> DVector<T> {
        let mut ret = DVector::zeros(hist.iter().map(|mat| mat.ncols()).sum());

        let mut sum = [T::zero(), T::zero(), T::zero()];
        for &(index, ref distance) in search_res {
            if distance.is_zero() {
                continue;
            }

            let weight = distance.clone().recip();
            let mut offset = 0;
            for (hist, sum) in hist.iter().zip(&mut sum) {
                for (i, elem) in hist.row(index).iter().enumerate() {
                    let value = elem.clone() * weight.clone();
                    *sum += value.clone();
                    ret[offset + i] += value;
                }
                offset += hist.ncols();
            }
        }

        let sum: [T; 3] = array::from_fn(|index| convert::<_, T>(100.) / sum[index].clone());

        ret.columns_range_mut(0..hist[0].ncols())
            .apply(|elem| *elem *= sum[0].clone());
        ret.columns_range_mut(hist[0].ncols()..)
            .columns_range_mut(..hist[1].ncols())
            .apply(|elem| *elem *= sum[1].clone());
        ret.columns_range_mut(hist[1].ncols()..)
            .apply(|elem| *elem *= sum[2].clone());

        ret
    }
}

impl<'a, 'b, T, I, S, V, N> Feature<PointCloud<I>, PointCloud<DVector<T>>, S, SearchType<T>>
    for FpfhEstimation<V>
where
    T: RealField + ToPrimitive,
    I: Point<Data = T> + 'a,
    S: Search<'a, I>,
    V: Deref<Target = [N]>,
    N: Normal<Data = T> + 'b,
{
    fn compute(
        &self,
        input: &PointCloud<I>,
        search: &S,
        search_param: SearchType<T>,
    ) -> PointCloud<DVector<T>> {
        let mut result = Vec::new();

        let (indices, hist) = self.compute_spfh(input, search, search_param.clone());

        let mut bounded = true;
        let storage = if input.is_bounded() {
            { input.iter() }
                .map(|point| {
                    search.search(point.coords(), search_param.clone(), &mut result);
                    if result.is_empty() {
                        bounded = false;
                        DVector::zeros(hist.iter().map(|mat| mat.ncols()).sum())
                    } else {
                        for (index, _) in result.iter_mut() {
                            *index = indices[*index];
                        }
                        self.weight_spfh(&hist, &result)
                    }
                })
                .collect::<Vec<_>>()
        } else {
            { input.iter() }
                .map(|point| {
                    if !point.is_finite() {
                        bounded = false;
                        return DVector::zeros(hist.iter().map(|mat| mat.ncols()).sum());
                    }
                    search.search(point.coords(), search_param.clone(), &mut result);
                    if result.is_empty() {
                        bounded = false;
                        DVector::zeros(hist.iter().map(|mat| mat.ncols()).sum())
                    } else {
                        for (index, _) in result.iter_mut() {
                            *index = indices[*index];
                        }
                        self.weight_spfh(&hist, &result)
                    }
                })
                .collect::<Vec<_>>()
        };

        unsafe { PointCloud::from_raw_parts(storage, input.width(), bounded) }
    }
}
