use std::collections::BinaryHeap;

#[derive(Debug, Copy, Clone)]
struct Node<K, V> {
    key: K,
    value: V,
}

impl<K: PartialEq, V: PartialEq> PartialEq for Node<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.value == other.value
    }
}

impl<K: PartialEq, V: PartialEq> Eq for Node<K, V> {}

impl<K: PartialOrd, V: PartialOrd> PartialOrd for Node<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.key.partial_cmp(&other.key) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.value.partial_cmp(&other.value)
    }
}

impl<K: PartialOrd, V: PartialOrd> Ord for Node<K, V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.key.partial_cmp(&other.key) {
            Some(std::cmp::Ordering::Equal) | None => {}
            Some(ord) => return ord,
        }
        self.value.partial_cmp(&other.value).unwrap()
    }
}

pub trait ResultSet {
    type Key;
    type Value;

    fn push(&mut self, key: Self::Key, value: Self::Value);

    fn is_full(&self) -> bool;

    fn max_key(&self) -> Option<&Self::Key>;
}

pub struct KnnResultSet<K, V> {
    data: BinaryHeap<Node<K, V>>,
    num: usize,
}

impl<K: PartialOrd, V: PartialOrd> KnnResultSet<K, V> {
    pub fn new(num: usize) -> Self {
        KnnResultSet {
            data: BinaryHeap::with_capacity(128),
            num,
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.data.iter().map(|node| (&node.key, &node.value))
    }
}

impl<K: PartialOrd, V: PartialOrd> IntoIterator for KnnResultSet<K, V> {
    type Item = (K, V);

    type IntoIter = impl Iterator<Item = (K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter().map(|node| (node.key, node.value))
    }
}

impl<K: PartialOrd, V: PartialOrd> ResultSet for KnnResultSet<K, V> {
    type Key = K;
    type Value = V;

    fn push(&mut self, key: K, value: V) {
        if self.max_key() <= Some(&key) {
            return;
        }

        if self.is_full() {
            self.data.pop();
        }

        self.data.push(Node { key, value });
    }

    fn is_full(&self) -> bool {
        self.data.len() >= self.num
    }

    fn max_key(&self) -> Option<&K> {
        self.data.peek().map(|node| &node.key)
    }
}

pub struct RadiusResultSet<K, V> {
    data: Vec<Node<K, V>>,
    radius: K,
}

impl<K: PartialOrd, V: PartialOrd> RadiusResultSet<K, V> {
    pub fn new(radius: K) -> Self {
        RadiusResultSet {
            data: Vec::with_capacity(128),
            radius,
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.data.iter().map(|node| (&node.key, &node.value))
    }
}

impl<K: PartialOrd, V: PartialOrd> IntoIterator for RadiusResultSet<K, V> {
    type Item = (K, V);

    type IntoIter = impl Iterator<Item = (K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter().map(|node| (node.key, node.value))
    }
}

impl<K: PartialOrd, V: PartialOrd> ResultSet for RadiusResultSet<K, V> {
    type Key = K;
    type Value = V;

    fn push(&mut self, key: K, value: V) {
        if key < self.radius {
            self.data.push(Node { key, value });
        }
    }

    fn is_full(&self) -> bool {
        true
    }

    fn max_key(&self) -> Option<&K> {
        Some(&self.radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_traits() {
        let node1 = Node {
            key: 0.0f32,
            value: 0.0,
        };
        let node2 = Node {
            key: 1.0f32,
            value: 1.0,
        };
        assert!(node1.cmp(&node2) == std::cmp::Ordering::Less);
    }
}
