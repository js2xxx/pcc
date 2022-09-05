#![feature(associated_type_defaults)]
#![feature(const_type_id)]
#![feature(generic_associated_types)]
#![feature(macro_metavar_expr)]
#![feature(map_try_insert)]
#![feature(type_alias_impl_trait)]
#![feature(unzip_option)]

use nalgebra::{Matrix3, RealField, Vector3, Vector4};

pub mod feature;
pub mod filter;
pub mod point;
pub mod point_cloud;
pub mod range_image;
pub mod search;

pub fn cov_matrix<'a, T, Iter>(coords: Iter) -> Option<Matrix3<T>>
where
    T: 'a + RealField,
    Iter: Iterator<Item = &'a Vector4<T>>,
{
    let mut cov = Matrix3::zeros();
    let mut mean = Vector3::zeros();
    let mut beta = T::zero();
    let mut num = 0;

    for coords in coords.map(|coords| {
        nalgebra::Point3::from_homogeneous(coords.clone())
            .unwrap()
            .coords
    }) {
        let alpha = T::one() - beta.clone();
        cov.syger(alpha.clone(), &coords, &coords, beta.clone());
        mean = coords * alpha.clone() + mean * beta.clone();
        beta = (T::one() + alpha).recip();
        num += 1;
    }

    (num >= 3).then(|| {
        cov.syger(-T::one(), &mean, &mean, T::one());
        cov
    })
}

pub fn normal<'a, T, Iter>(coords: Iter, viewpoint: &Vector4<T>) -> Option<(Vector4<T>, T)>
where
    T: 'a + RealField,
    Iter: Iterator<Item = &'a Vector4<T>>,
{
    let se = cov_matrix(coords)?.symmetric_eigen();
    let index = se.eigenvalues.imin();
    let mut normal = se.eigenvectors.column(index).into_owned();
    let curvature = se.eigenvalues[index].clone() / se.eigenvalues.sum();

    if normal.dot(&viewpoint.xyz()) < T::zero() {
        normal.neg_mut();
    }

    Some((normal.insert_row(3, T::zero()), curvature))
}
