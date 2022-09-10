use nalgebra::{RealField, Scalar, Vector4};
use pcc_common::{
    feature::Feature,
    point::Point,
    point_cloud::PointCloud,
    search::{Search, SearchType},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Normal<T: Scalar> {
    pub viewpoint: Vector4<T>,
}

impl<T: Scalar> Normal<T> {
    pub fn new(viewpoint: Vector4<T>) -> Self {
        Normal { viewpoint }
    }
}

impl<'a, T, I, O, S> Feature<&'a PointCloud<I>, PointCloud<O>, S, SearchType<T>> for Normal<T>
where
    T: RealField,
    I: Point<Data = T> + 'a,
    S: Search<'a, I>,
    O: pcc_common::point::Normal<Data = T>,
{
    fn compute(
        &self,
        input: &'a PointCloud<I>,
        search: S,
        search_param: SearchType<T>,
    ) -> PointCloud<O> {
        let mut result = Vec::new();
        if input.is_bounded() {
            let storage = { input.iter() }
                .map(|point| {
                    search.search(point.coords(), search_param.clone(), &mut result);
                    let res = pcc_common::normal(
                        result
                            .iter()
                            .map(|&(index, _)| search.input()[index].coords()),
                        &self.viewpoint,
                    )
                    .map(|(normal, curvature)| {
                        O::default().with_normal(normal).with_curvature(curvature)
                    });
                    res.unwrap_or_default()
                })
                .collect::<Vec<_>>();
            PointCloud::from_vec(storage, input.width())
        } else {
            let storage = { input.iter() }
                .map(|point| {
                    if !point.is_finite() {
                        return Default::default();
                    }
                    search.search(point.coords(), search_param.clone(), &mut result);
                    let res = pcc_common::normal(
                        result
                            .iter()
                            .map(|&(index, _)| search.input()[index].coords()),
                        &self.viewpoint,
                    )
                    .map(|(normal, curvature)| {
                        O::default().with_normal(normal).with_curvature(curvature)
                    });
                    res.unwrap_or_default()
                })
                .collect::<Vec<_>>();
            PointCloud::from_vec(storage, input.width())
        }
    }
}
