use nalgebra::{RealField, Scalar};
use pcc_common::{
    filter::ApproxFilter,
    point_cloud::PointCloud,
    points::{Point3Infoed, PointIntensity},
};
use pcc_kdtree::{KdTree, RadiusResultSet};

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
        (-x.clone() * x / ((T::one() + T::one()) * sigma.clone() * sigma)).exp()
    }

    fn compute_intensity<'a, I: 'a + PointIntensity<T>, Iter>(
        &self,
        pivot: &Point3Infoed<T, I>,
        data: Iter,
    ) -> T
    where
        Iter: Iterator<Item = (&'a Point3Infoed<T, I>, T)>,
    {
        let (sum, weight) = data.fold(
            (T::zero(), T::zero()),
            |(sum, weight), (point, distance)| {
                let intensity = (**point.extra.intensity()).clone();
                let id = (intensity.clone() - (**pivot.extra.intensity()).clone()).abs();
                let w = Self::kernel(distance, self.sigma_d.clone())
                    * Self::kernel(id, self.sigma_r.clone());

                (sum + w.clone() * intensity, weight + w)
            },
        );
        sum / weight
    }
}

impl<T: RealField, I: PointIntensity<T> + Clone> ApproxFilter<PointCloud<Point3Infoed<T, I>>>
    for Bilateral<T>
{
    fn filter(&mut self, input: &PointCloud<Point3Infoed<T, I>>) -> PointCloud<Point3Infoed<T, I>> {
        let searcher = KdTree::new(input);

        let mut result = RadiusResultSet::new(self.sigma_d.clone() * (T::one() + T::one()));
        let mut output = input.clone();
        unsafe {
            for point in output.storage().iter_mut() {
                searcher.search_typed(&point.coords, &mut result);
                **point.extra.intensity_mut() = self.compute_intensity(
                    point,
                    { result.iter() }.map(|(distance, index)| (&input[*index], distance.clone())),
                );
            }
        }

        output
    }
}
