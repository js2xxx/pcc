use nalgebra::{RealField, Scalar, Vector3, Vector4};
use num::zero;
use pcc_common::{
    feature::Feature,
    point::{Normal, Point},
    point_cloud::PointCloud,
    search::{Search, SearchType},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryEstimation<T: Scalar, N> {
    pub angle_threshold: T,
    pub normals: N,
}

impl<T: Scalar, N> BoundaryEstimation<T, N> {
    pub fn new(angle_threshold: T, normals: N) -> Self {
        BoundaryEstimation {
            angle_threshold,
            normals,
        }
    }
}

impl<T: RealField, N> BoundaryEstimation<T, N> {
    fn boundary<'a, Iter>(&self, pivot: &Vector4<T>, coords: Iter, [u, v]: &[Vector3<T>; 2]) -> bool
    where
        Iter: Iterator<Item = &'a Vector4<T>>,
    {
        if pivot.iter().any(|x| !x.is_finite()) {
            return false;
        }

        let mut angles = coords
            .map(|coords| {
                let delta = (pivot - coords).xyz();
                if delta == Vector3::zeros() {
                    zero()
                } else {
                    (delta.dot(v)).atan2(delta.dot(u))
                }
            })
            .collect::<Vec<_>>();
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let diff = { angles.array_windows::<2>() }
            .fold(zero(), |acc, [a, b]| (b.clone() - a.clone()).max(acc));
        diff.max(T::two_pi() + angles.first().unwrap().clone() - angles.last().unwrap().clone())
            > self.angle_threshold
    }
}

impl<'a, 'b, T, I, S, V, N> Feature<PointCloud<I>, PointCloud<bool>, S, SearchType<T>>
    for BoundaryEstimation<T, V>
where
    T: RealField,
    I: Point<Data = T>,
    S: Search<'a, I>,
    V: Iterator<Item = &'b N> + Clone,
    N: Normal<Data = T> + 'b,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    fn compute(
        &self,
        input: &PointCloud<I>,
        search: &S,
        search_param: SearchType<T>,
    ) -> PointCloud<bool> {
        let mut result = Vec::new();
        let mut bounded = true;
        let storage = if input.is_bounded() {
            input
                .iter()
                .zip(self.normals.clone())
                .map(|(point, normal)| {
                    search.search(point.coords(), search_param.clone(), &mut result);
                    if result.is_empty() {
                        bounded = false;
                        return false;
                    }
                    let u =
                        { normal.normal().xyz() }.cross(&Vector3::from(rand::random::<[T; 3]>()));
                    let v = normal.normal().xyz().cross(&u);
                    self.boundary(
                        point.coords(),
                        result.iter().map(|&(index, _)| input[index].coords()),
                        &[u, v],
                    )
                })
                .collect::<Vec<_>>()
        } else {
            input
                .iter()
                .zip(self.normals.clone())
                .map(|(point, normal)| {
                    if !point.is_finite() {
                        bounded = false;
                        return false;
                    }
                    search.search(point.coords(), search_param.clone(), &mut result);
                    if result.is_empty() {
                        bounded = false;
                        return false;
                    }
                    let u =
                        { normal.normal().xyz() }.cross(&Vector3::from(rand::random::<[T; 3]>()));
                    let v = normal.normal().xyz().cross(&u);
                    self.boundary(
                        point.coords(),
                        result.iter().map(|&(index, _)| input[index].coords()),
                        &[u, v],
                    )
                })
                .collect::<Vec<_>>()
        };
        unsafe { PointCloud::from_raw_parts(storage, input.width(), bounded) }
    }
}
