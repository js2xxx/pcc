use pcc_common::filter::Filter;

pub struct Simple<'a, T> {
    constraint: Box<dyn FnMut(&T) -> bool + 'a>,
}

impl<'a, T> Simple<'a, T> {
    pub fn new(constraint: impl Fn(&T) -> bool + 'a) -> Self {
        Simple {
            constraint: Box::new(constraint),
        }
    }
}

impl<'a, T: Clone> Filter<[T]> for Simple<'a, T> {
    fn filter_indices(&mut self, input: &[T]) -> Vec<usize> {
        let mut indices = (0..input.len()).collect::<Vec<_>>();
        indices.retain(|&index| (self.constraint)(&input[index]));
        indices
    }

    fn filter_all_indices(&mut self, input: &[T]) -> (Vec<usize>, Vec<usize>) {
        let mut indices = (0..input.len()).collect::<Vec<_>>();
        let mut removed = Vec::with_capacity(indices.len());
        indices.retain(|&index| {
            let ret = (self.constraint)(&input[index]);
            if !ret {
                removed.push(index)
            }
            ret
        });
        (indices, removed)
    }
}
