mod base;
pub mod circle;
pub mod cone;
pub mod cylinder;
pub mod line;
pub mod plane;
pub mod sphere;

pub use self::base::{Arrsac, Sac};

#[cfg(test)]
mod tests {
    use nalgebra::matrix;
    use sample_consensus::Consensus;

    use crate::{base::Arrsac, line::LineEstimator};

    #[test]
    fn test_line() {
        let mut sac = Arrsac::new(1., rand::thread_rng());
        let points = [
            matrix![0.; 0.; 0.; 1.],
            matrix![1.; 1.; 1.; 1.],
            matrix![2.; 2.; 3.; 1.],
            matrix![-1.; 4.; -2.; 1.],
            matrix![5.; 5.; 4.; 1.],
            matrix![8.; 4.; 3.; 1.],
            matrix![15.; 13.; 14.; 1.],
            matrix![-7.; 6.; 7.; 1.],
        ];
        let (_model, inliners) = sac
            .model_inliers(&LineEstimator, points.into_iter())
            .unwrap();
        assert_eq!(inliners, vec![0, 1, 2, 4, 6]);
    }
}
