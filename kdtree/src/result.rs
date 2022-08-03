// use std::collections::BinaryHeap;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Node<K, V> {
    key: K,
    value: V,
}

impl<K: PartialOrd, V: PartialOrd> PartialOrd for Node<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.key.partial_cmp(&other.key) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.value.partial_cmp(&other.value)
    }
}

impl<K: Ord, V: PartialOrd + Eq> Ord for Node<K, V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.key.cmp(&other.key) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        self.value.partial_cmp(&other.value).unwrap()
    }
}

pub trait ResultSet<K, V> {
    fn push(&mut self, key: K, value: V);

    fn is_full(&self) -> bool;

    fn max_key(&self) -> Option<&K>;
}

// pub struct ResultSet<K, V>(BinaryHeap<Node<K, V>>);

// impl<K: Ord, V: PartialOrd + Eq> ResultSet<K, V> {
//     pub fn new() -> Self {
//         ResultSet(BinaryHeap::new())
//     }

//     pub fn len(&self) -> usize {
//         self.0.len()
//     }

//     #[must_use]
//     pub fn is_empty(&self) -> bool {
//         self.0.is_empty()
//     }

//     pub fn peek(&self) -> Option<&K> {
//         self.0.peek().map(|node| &node.key)
//     }

//     pub fn modify_top<F>(&mut self, mut f: F)
//     where
//         F: FnMut(&mut K, &mut V),
//     {
//         if let Some(mut pm) = self.0.peek_mut() {
//             let Node { key, value } = &mut *pm;
//             f(key, value)
//         }
//     }

//     pub fn push(&mut self, key: K, value: V) {
//         self.0.push(Node { key, value });
//     }

//     pub fn pop(&mut self) -> Option<(K, V)> {
//         let Node { key, value } = self.0.pop()?;
//         Some((key, value))
//     }

//     pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
//         self.0.iter().map(|node| (&node.key, &node.value))
//     }
// }

// impl<K: Ord, V: Ord> Default for ResultSet<K, V> {
//     fn default() -> Self {
//         Self::new()
//     }
// }

// impl<K: Ord, V: Ord> IntoIterator for ResultSet<K, V> {
//     type Item = (K, V);

//     type IntoIter = impl Iterator<Item = (K, V)>;

//     fn into_iter(self) -> Self::IntoIter {
//         self.0.into_iter().map(|node| (node.key, node.value))
//     }
// }
