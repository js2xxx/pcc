mod neighbors;

use nalgebra::RealField;
use num::ToPrimitive;
use pcc_common::{point::Point, point_cloud::PointCloud, search::Search};
pub use pcc_kdtree::*;
pub use pcc_octree::*;

pub use self::neighbors::*;

#[inline]
pub fn __searcher<'a, 'b, T, P>(
    input: &'a PointCloud<P>,
    epsilon: P::Data,
    storage: &'b mut (Option<OrganizedNeighbor<'a, P>>, Option<KdTree<'a, P>>),
) -> &'b dyn Search<'a, P>
where
    P: Point<Data = T>,
    T: RealField + ToPrimitive,
{
    let org_neigh: Option<&dyn Search<'a, P>> = if input.width() > 1 {
        OrganizedNeighbor::new(input, epsilon).map(|x| storage.0.insert(x) as _)
    } else {
        None
    };
    org_neigh.unwrap_or_else(|| storage.1.insert(KdTree::new(input)))
}

#[macro_export]
macro_rules! searcher {
    ($ident:ident in $input:ident, $epsilon:expr) => {
        let mut __storage = Default::default();
        let $ident = $crate::__searcher($input, $epsilon, &mut __storage);
    };
}
