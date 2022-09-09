use nalgebra::{convert, RealField, Vector3, Vector4};
use pcc_common::{
    feature::Feature,
    point::Point,
    point_cloud::{AsPointCloud, PointCloud},
    search::{Search, SearchType},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MomentInvariant;

impl MomentInvariant {
    fn point_mi<T, P>(
        result: &[(usize, T)],
        input: &PointCloud<P>,
        centroid: &Vector4<T>,
    ) -> Vector3<T>
    where
        T: RealField,
        P: Point<Data = T>,
    {
        let mut sqr = Vector3::zeros();
        let mut dot = Vector3::zeros();
        for &(index, _) in result.iter() {
            let diff = input[index].coords() - centroid;
            sqr += diff.xyz().component_mul(&diff.xyz());
            dot += diff.yzx().component_mul(&diff.zxy());
        }
        let x0 = sqr.sum();
        let x1 = sqr.dot(&sqr.yzx()) - dot.dot(&dot.yzx());
        let x2 = sqr.product() + dot.product() * convert(2.) - dot.component_mul(&dot).dot(&sqr);
        Vector3::new(x0, x1, x2)
    }
}

impl<'a, T, P, S> Feature<&'a PointCloud<P>, Option<PointCloud<Vector3<T>>>, S, SearchType<T>>
    for MomentInvariant
where
    T: RealField,
    P: Point<Data = T>,
    S: Search<'a, P>,
{
    fn compute(
        &self,
        input: &'a PointCloud<P>,
        search: S,
        search_param: SearchType<T>,
    ) -> Option<PointCloud<Vector3<T>>> {
        let centroid = input.centroid_coords().0?;

        let mut result = Vec::new();
        let mut bounded = input.is_bounded();
        let storage = if bounded {
            let iter = input.iter().map(|point| {
                search.search(point.coords(), search_param.clone(), &mut result);
                if result.is_empty() {
                    bounded = false;
                    Vector3::zeros()
                } else {
                    Self::point_mi(&result, input, &centroid)
                }
            });
            iter.collect::<Vec<_>>()
        } else {
            let iter = input.iter().map(|point| {
                if !point.is_finite() {
                    return Vector3::zeros();
                }
                search.search(point.coords(), search_param.clone(), &mut result);
                if result.is_empty() {
                    Vector3::zeros()
                } else {
                    Self::point_mi(&result, input, &centroid)
                }
            });
            iter.collect::<Vec<_>>()
        };
        Some(unsafe { PointCloud::from_raw_parts(storage, input.width(), bounded) })
    }
}
