pub trait Feature<I, O, S, P> {
    fn compute(&self, input: &I, search: &S, search_param: P) -> O;
}
