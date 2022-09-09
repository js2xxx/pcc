use std::collections::{HashMap, VecDeque};

use nalgebra::{convert, DVector, RealField, Unit, Vector3};
use num::ToPrimitive;
use pcc_common::{
    feature::Feature,
    point::{Normal, Point},
    point_cloud::PointCloud,
    search::{Search, SearchType},
};

use crate::HIST_MAX;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub(crate) struct PfhPair<T> {
    pub alpha: T,
    pub phi: T,
    pub theta: T,
    pub distance: T,
}

impl<T: RealField> PfhPair<T> {
    pub fn try_new([p1, n1]: &[Vector3<T>; 2], [p2, n2]: &[Vector3<T>; 2]) -> Option<PfhPair<T>> {
        let (delta, distance) = Unit::try_new_and_get(p2 - p1, T::zero())?;

        let u = n1;
        let v = u.cross(&delta);
        let w = u.cross(&v);

        Some(PfhPair {
            alpha: v.dot(n2),
            phi: u.dot(&delta),
            theta: w.dot(n2).atan2(u.dot(n2)),
            distance,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Pfh {
    pub cache_len: usize,
    pub subdivision: usize,
}

impl Pfh {
    pub fn new(cache_size: usize, subdivision: usize) -> Self {
        Pfh {
            cache_len: cache_size,
            subdivision,
        }
    }

    fn pfh<T, P, N>(
        &self,
        indices: &[(usize, T)],
        points: &[P],
        normals: &[N],
        cache: &mut HashMap<(usize, usize), PfhPair<T>>,
        cached_keys: &mut VecDeque<(usize, usize)>,
    ) -> DVector<T>
    where
        T: RealField + ToPrimitive,
        P: Point<Data = T>,
        N: Normal<Data = T>,
    {
        let num: T = convert(self.subdivision as f64);
        let inc =
            convert::<_, T>(HIST_MAX) / convert(((indices.len() - 1) * indices.len() / 2) as f64);

        let mut count = vec![T::zero(); self.subdivision * self.subdivision * self.subdivision];
        for (i, j) in indices
            .iter()
            .enumerate()
            .flat_map(|(count, &(i, _))| indices.iter().take(count).map(move |&(j, _)| (i, j)))
        {
            let (pair, new) = {
                let mut new = false;
                let pair = (self.cache_len > 0)
                    .then(|| cache.get(&(i, j)).cloned())
                    .flatten();
                let pair = pair.or_else(|| {
                    new = true;
                    PfhPair::try_new(
                        &[points[i].coords().xyz(), normals[i].normal().xyz()],
                        &[points[j].coords().xyz(), normals[j].normal().xyz()],
                    )
                });
                match pair {
                    Some(pair) => (pair, new),
                    None => continue,
                }
            };
            let (index, _) = {
                let data = [
                    ((pair.theta.clone() + T::pi()) / T::two_pi() * num.clone()),
                    ((pair.alpha.clone() + T::one()) / convert(2.) * num.clone()),
                    ((pair.phi.clone() + T::one()) / convert(2.) * num.clone()),
                ];
                data.into_iter().fold((0, 1), |(index, weight), data| {
                    let data = { data.clamp(T::zero(), num.clone()).floor() }
                        .to_usize()
                        .unwrap();
                    (index + weight * data, weight * self.subdivision)
                })
            };
            count[index] += inc.clone();

            if self.cache_len > 0 && new {
                cache.insert((i, j), pair);
                cached_keys.push_back((i, j));

                if cached_keys.len() > self.cache_len {
                    let old = cached_keys.pop_front().unwrap();
                    cache.remove(&old);
                }
            }
        }

        count.into()
    }
}

impl<'a, 'b, T, I, S, N>
    Feature<(&'a PointCloud<I>, &'b PointCloud<N>), PointCloud<DVector<T>>, S, SearchType<T>>
    for Pfh
where
    T: RealField + ToPrimitive,
    I: Point<Data = T> + 'a,
    S: Search<'a, I>,
    N: Normal<Data = T> + 'b,
{
    fn compute(
        &self,
        (input, normals): (&'a PointCloud<I>, &'b PointCloud<N>),
        search: S,
        search_param: SearchType<T>,
    ) -> PointCloud<DVector<T>> {
        let mut result = Vec::new();
        let mut bounded = true;

        let mut cache = HashMap::new();
        let mut cached_keys = VecDeque::new();

        let storage = if input.is_bounded() {
            input
                .iter()
                .map(|point| {
                    search.search(point.coords(), search_param.clone(), &mut result);
                    if result.is_empty() {
                        bounded = false;
                        return DVector::from(Vec::new());
                    }
                    self.pfh(
                        &result,
                        search.input(),
                        normals,
                        &mut cache,
                        &mut cached_keys,
                    )
                })
                .collect::<Vec<_>>()
        } else {
            input
                .iter()
                .map(|point| {
                    if !point.is_finite() {
                        bounded = false;
                        return DVector::from(Vec::new());
                    }
                    search.search(point.coords(), search_param.clone(), &mut result);
                    if result.is_empty() {
                        bounded = false;
                        return DVector::from(Vec::new());
                    }
                    self.pfh(
                        &result,
                        search.input(),
                        normals,
                        &mut cache,
                        &mut cached_keys,
                    )
                })
                .collect::<Vec<_>>()
        };
        unsafe { PointCloud::from_raw_parts(storage, input.width(), bounded) }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_flat_map() {
        let options = [Some(123), None, Some(3425)];
        let result = options
            .into_iter()
            .flat_map(|option| option.into_iter())
            .collect::<Vec<_>>();
        assert_eq!(result, vec![123, 3425]);
    }
}
