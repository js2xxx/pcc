use nalgebra::{ComplexField, SVector, Scalar};

pub trait Centroid {
    type Accumulator;

    fn default_builder() -> CentroidBuilder<Self>
    where
        Self: Sized,
        Self::Accumulator: Default,
    {
        CentroidBuilder::default()
    }

    fn builder(accum: Self::Accumulator) -> CentroidBuilder<Self>
    where
        Self: Sized,
    {
        CentroidBuilder::new(accum)
    }

    fn accumulate(&self, accum: &mut Self::Accumulator);

    fn compute(accum: Self::Accumulator, num: usize) -> Self;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CentroidBuilder<T: Centroid> {
    accum: T::Accumulator,
    num: usize,
}

impl<T: Centroid> Default for CentroidBuilder<T>
where
    T::Accumulator: Default,
{
    fn default() -> Self {
        Self::new(T::Accumulator::default())
    }
}

impl<T: Centroid> CentroidBuilder<T> {
    pub fn new(accum: T::Accumulator) -> Self {
        CentroidBuilder { accum, num: 0 }
    }

    pub fn accumulate(&mut self, obj: &T) {
        obj.accumulate(&mut self.accum);
        self.num += 1;
    }

    pub fn num(&self) -> usize {
        self.num
    }

    pub fn compute(self) -> Option<T> {
        (self.num > 0).then(|| T::compute(self.accum, self.num))
    }
}

impl<T: Scalar + ComplexField<RealField = T>, const D: usize> Centroid for SVector<T, D> {
    type Accumulator = Self;

    fn accumulate(&self, accum: &mut Self::Accumulator) {
        *accum += self;
    }

    fn compute(accum: Self::Accumulator, num: usize) -> Self {
        accum / T::from_usize(num).unwrap()
    }
}

macro_rules! impl_tuples {
    ($($id:ident),*) => {
        impl<$($id : Centroid),*> Centroid for ($($id),*) {
            type Accumulator = ($($id ::Accumulator),*);

            fn accumulate(&self, _accum: &mut Self::Accumulator) {
                $(${ignore(id)} self. ${index()} .accumulate(&mut _accum. ${index()}));*
            }

            #[allow(clippy::unused_unit)]
            fn compute(_accum: Self::Accumulator, _num: usize) -> Self {
                ($(<$id as Centroid>::compute(_accum. ${index()}, _num)),*)
            }
        }
    };
}

impl_tuples!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_tuples!(A, B, C, D, E, F, G, H, I, J, K);
impl_tuples!(A, B, C, D, E, F, G, H, I, J);
impl_tuples!(A, B, C, D, E, F, G, H, I);
impl_tuples!(A, B, C, D, E, F, G, H);
impl_tuples!(A, B, C, D, E, F, G);
impl_tuples!(A, B, C, D, E, F);
impl_tuples!(A, B, C, D, E);
impl_tuples!(A, B, C, D);
impl_tuples!(A, B, C);
impl_tuples!(A, B);
impl_tuples!();
