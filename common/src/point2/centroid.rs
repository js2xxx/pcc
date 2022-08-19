pub trait Centroid {
    type Accumulator;
    type Result;

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

    fn compute(accum: Self::Accumulator, num: usize) -> Self::Result;
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

    pub fn compute(self) -> Option<T::Result> {
        (self.num > 0).then(|| T::compute(self.accum, self.num))
    }
}
