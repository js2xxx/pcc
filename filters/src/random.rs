use std::mem;

use pcc_common::filter::Filter;
use rand::{rngs::ThreadRng, RngCore};

pub struct Random<R: RngCore = ThreadRng> {
    pub rng: R,
    pub select_num: usize,
}

impl<R: RngCore> Random<R> {
    pub fn new(rng: R, select_num: usize) -> Random<R> {
        Random { rng, select_num }
    }
}

impl<R: RngCore, T> Filter<[T]> for Random<R> {
    fn filter_indices(&mut self, input: &[T]) -> Vec<usize> {
        if input.len() <= self.select_num {
            (0..input.len()).collect()
        } else {
            let mut indices = (0..self.select_num).collect::<Vec<_>>();

            for index in self.select_num..input.len() {
                let prob = self.rng.next_u64() as usize % index;
                if prob < self.select_num {
                    indices[prob] = index;
                }
            }

            indices
        }
    }

    fn filter_all_indices(&mut self, input: &[T]) -> (Vec<usize>, Vec<usize>) {
        if input.len() <= self.select_num {
            ((0..input.len()).collect(), Vec::new())
        } else {
            let mut indices = (0..self.select_num).collect::<Vec<_>>();
            let mut removed = (self.select_num..input.len()).collect::<Vec<_>>();

            for index in removed.iter_mut() {
                let prob = self.rng.next_u64() as usize % *index;
                if prob < self.select_num {
                    mem::swap(&mut indices[prob], index);
                }
            }

            (indices, removed)
        }
    }
}
