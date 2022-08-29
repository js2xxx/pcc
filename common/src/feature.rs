pub trait Feature<I, O> {
    fn compute(&self, input: &I) -> O;
}
