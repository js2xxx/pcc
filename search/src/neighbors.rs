use std::iter;

use nalgebra::{ComplexField, Matrix3, Matrix3x4, RealField, Vector2, Vector4};
use num::{zero, FromPrimitive, ToPrimitive};
use pcc_common::{
    point::Point,
    point_cloud::PointCloud,
    search::{Search, SearchType},
};
use pcc_kdtree::{KnnResultSet, ResultSet};

pub struct OrganizedNeighbor<'a, P: Point> {
    point_cloud: &'a PointCloud<P>,
    proj_matrix: Matrix3x4<P::Data>,
    kr: Matrix3<P::Data>,
    kr_krt: Matrix3<P::Data>,
}

impl<'a, P> OrganizedNeighbor<'a, P>
where
    P: Point,
    P::Data: RealField,
{
    pub fn new(point_cloud: &'a PointCloud<P>, epsilon: P::Data) -> Option<Self> {
        let (proj_matrix, residual) = point_cloud.proj_matrix();
        (residual <= P::Data::from_usize(point_cloud.len()).unwrap() * epsilon).then(|| {
            let kr = proj_matrix.fixed_slice::<3, 3>(0, 0).into_owned();
            let kr_krt = &kr * kr.transpose();
            OrganizedNeighbor {
                point_cloud,
                proj_matrix,
                kr,
                kr_krt,
            }
        })
    }

    pub fn project(&self, coords: &Vector4<P::Data>) -> Option<Vector2<P::Data>> {
        let p = &self.kr * coords.xyz() + self.proj_matrix.column(3);
        nalgebra::Point2::from_homogeneous(p).map(|p| p.coords)
    }
}

impl<'a, P> OrganizedNeighbor<'a, P>
where
    P: Point,
    P::Data: RealField + ToPrimitive,
{
    fn search_box(&self, pivot: &Vector4<P::Data>, radius_sqr: P::Data) -> [usize; 4] {
        let pp = &self.kr * pivot.xyz() + self.proj_matrix.column(3);

        let a = radius_sqr.clone() * self.kr_krt[8].clone() - pp.z.clone() * pp.z.clone();
        let b = radius_sqr.clone() * self.kr_krt[7].clone() - pp.y.clone() * pp.z.clone();
        let c = radius_sqr.clone() * self.kr_krt[4].clone() - pp.y.clone() * pp.y.clone();

        let d = b.clone() * b.clone() - a.clone() * c;
        let ylimit = self.point_cloud.height() - 1;
        let [ymin, ymax] = if d >= zero() {
            let y1 = (b.clone() - d.clone().sqrt()) / a.clone();
            let y2 = (b + d.sqrt()) / a.clone();

            let min = y1
                .clone()
                .floor()
                .min(y2.clone().floor())
                .to_usize()
                .unwrap();
            let max = y1.ceil().min(y2.ceil()).to_usize().unwrap();

            [min.clamp(0, ylimit), max.clamp(0, ylimit)]
        } else {
            [0, ylimit]
        };

        let b = radius_sqr.clone() * self.kr_krt[6].clone() - pp.x.clone() * pp.z.clone();
        let c = radius_sqr * self.kr_krt[0].clone() - pp.x.clone() * pp.x.clone();
        let d = b.clone() * b.clone() - a.clone() * c;
        let xlimit = self.point_cloud.width() - 1;
        let [xmin, xmax] = if d >= zero() {
            let x1 = (b.clone() - d.clone().sqrt()) / a.clone();
            let x2 = (b + d.sqrt()) / a;

            let min = x1
                .clone()
                .floor()
                .min(x2.clone().floor())
                .to_usize()
                .unwrap();
            let max = x1.ceil().min(x2.ceil()).to_usize().unwrap();

            [min.clamp(0, xlimit), max.clamp(0, xlimit)]
        } else {
            [0, xlimit]
        };

        [xmin, xmax, ymin, ymax]
    }

    pub fn radius_search(
        &self,
        pivot: &Vector4<P::Data>,
        radius: P::Data,
        result: &mut Vec<(usize, P::Data)>,
    ) {
        result.clear();

        let [xmin, xmax, ymin, ymax] = self.search_box(pivot, radius.clone() * radius.clone());
        for x in xmin..=xmax {
            for y in ymin..=ymax {
                let index = self.point_cloud.width() * y + x;
                let distance = (self.point_cloud[index].coords() - pivot).norm();
                if distance <= radius {
                    result.push((index, distance));
                }
            }
        }
    }

    pub fn knn_search(
        &self,
        pivot: &Vector4<P::Data>,
        n: usize,
        result: &mut Vec<(usize, P::Data)>,
    ) {
        result.clear();

        let mut rr = KnnResultSet::new(n);

        let [[x, y]] = self.project(pivot).unwrap().map(|x| x.round()).data.0;

        let (mut wxmin, mut wxmax) = (0, self.point_cloud.width() - 1);
        let (mut wymin, mut wymax) = (0, self.point_cloud.height() - 1);

        let (mut vxmin, mut vymin) = (
            x.clamp(zero(), P::Data::from_usize(wxmax).unwrap())
                .to_usize()
                .unwrap(),
            y.clamp(zero(), P::Data::from_usize(wymax).unwrap())
                .to_usize()
                .unwrap(),
        );
        let (mut vxmax, mut vymax) = (vxmin, vymin);

        {
            let index = vymin * self.point_cloud.width() + vxmin;
            let distance = (self.point_cloud[index].coords() - pivot).norm();
            rr.push(distance, index);
            if rr.is_full() {
                [wxmin, wxmax, wymin, wymax] =
                    self.search_box(pivot, rr.max_key().unwrap().clone());
            }
        }

        loop {
            let mut points = {
                let top = (vxmax < wxmax && vymax < wymax)
                    .then(|| (vxmin..=(vxmax + 1)).zip(iter::repeat(vymax + 1)))
                    .into_iter()
                    .flatten();
                let bottom = (vxmin > wxmin && vymin > wymin)
                    .then(|| ((vxmin - 1)..=vxmax).zip(iter::repeat(vymin - 1)))
                    .into_iter()
                    .flatten();
                let left = (vxmin > wxmin && vymax < wymax)
                    .then(|| iter::repeat(vxmin - 1).zip(vymin..=(vymax + 1)))
                    .into_iter()
                    .flatten();
                let right = (vxmax < wxmax && vymin > wymin)
                    .then(|| iter::repeat(vxmax + 1).zip((vymin - 1)..=vymax))
                    .into_iter()
                    .flatten();
                top.chain(bottom).chain(left).chain(right).peekable()
            };
            if points.peek().is_none() {
                result.extend(rr.into_iter().map(|(d, v)| (v, d)));
                break;
            }

            for (x, y) in points {
                let index = y * self.point_cloud.width() + x;
                let distance = (self.point_cloud[index].coords() - pivot).norm();
                rr.push(distance, index);
                if rr.is_full() {
                    [wxmin, wxmax, wymin, wymax] =
                        self.search_box(pivot, rr.max_key().unwrap().clone());
                }
            }
            vxmin = vxmin.saturating_sub(1).max(wxmin);
            vxmax = (vxmax + 1).min(wxmax);
            vymin = vymin.saturating_sub(1).max(wymin);
            vymax = (vymax + 1).min(wymax);
        }
    }
}

impl<'a, P> Search<'a, P> for OrganizedNeighbor<'a, P>
where
    P: Point,
    P::Data: RealField + ToPrimitive,
{
    fn point_cloud(&self) -> &'a PointCloud<P> {
        self.point_cloud
    }

    fn search(
        &self,
        pivot: &Vector4<<P>::Data>,
        ty: SearchType<<P>::Data>,
        result: &mut Vec<(usize, <P>::Data)>,
    ) {
        match ty {
            SearchType::Knn(n) => self.knn_search(pivot, n, result),
            SearchType::Radius(radius) => self.radius_search(pivot, radius, result),
        }
    }
}
