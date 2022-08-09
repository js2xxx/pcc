pub trait Filter<'a, T: 'a> {
    fn append_constraint<C>(&mut self, constraint: C)
    where
        C: FnMut(&'a T) -> bool;

    fn constraint<C>(&mut self, constraint: C) -> &mut Self
    where
        C: FnMut(&'a T) -> bool,
    {
        self.append_constraint(constraint);
        self
    }

    fn result(&self) -> &[usize];

    /// This function may return less than the exact result of the removed
    /// indices of points. The empty slices returned are often considered not
    /// stored by the filter.
    fn removed(&self) -> &[usize] {
        &[]
    }
}
