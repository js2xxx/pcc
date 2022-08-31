use nalgebra::{convert, RealField, Scalar};
use num::ToPrimitive;
use pcc_common::{
    filter::ApproxFilter, point::PointIntensity, point_cloud::PointCloud, search::SearchType,
};
use pcc_search::searcher;

/// NOTE: This function don't modify point coordinates. Instead, it recomputes
/// their intensities.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bilateral<T: Scalar> {
    pub sigma_d: T,
    pub sigma_r: T,
}

impl<T: Scalar> Bilateral<T> {
    pub fn new(sigma_d: T, sigma_r: T) -> Self {
        Bilateral { sigma_d, sigma_r }
    }
}

impl<T: RealField> Bilateral<T> {
    fn kernel(x: T, sigma: T) -> T {
        (-x.clone() * x / (convert::<_, T>(2.) * sigma.clone() * sigma)).exp()
    }

    fn compute_intensity<'a, P: 'a + PointIntensity<Data = T>, Iter>(
        &self,
        pivot: &P,
        data: Iter,
    ) -> T
    where
        Iter: Iterator<Item = (&'a P, T)>,
    {
        let (sum, weight) = data.fold(
            (T::zero(), T::zero()),
            |(sum, weight), (point, distance)| {
                let intensity = point.intensity();
                let id = (intensity.clone() - pivot.intensity()).abs();
                let w = Self::kernel(distance, self.sigma_d.clone())
                    * Self::kernel(id, self.sigma_r.clone());

                (sum + w.clone() * intensity, weight + w)
            },
        );
        sum / weight
    }
}

impl<T: RealField + ToPrimitive, P: PointIntensity<Data = T>> ApproxFilter<PointCloud<P>>
    for Bilateral<T>
{
    fn filter(&mut self, input: &PointCloud<P>) -> PointCloud<P> {
        searcher!(searcher in input, T::default_epsilon());

        let radius = self.sigma_d.clone() * convert(2.);
        let mut result = Vec::new();
        let mut output = input.clone();
        unsafe {
            for point in output.storage().iter_mut() {
                searcher.search(
                    point.coords(),
                    SearchType::Radius(radius.clone()),
                    &mut result,
                );
                point.set_intensity(self.compute_intensity(
                    point,
                    { result.iter() }.map(|(index, distance)| (&input[*index], distance.clone())),
                ));
            }
        }

        output
    }
}
