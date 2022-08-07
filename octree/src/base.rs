use std::{io, ptr::NonNull};

use crate::{iter::*, node::Node};

#[derive(Debug)]
pub struct OcTree<T> {
    root: Option<NonNull<Node<(), T>>>,
    depth: usize,
}

impl<T> OcTree<T> {
    pub fn new(depth: usize) -> Self {
        OcTree { root: None, depth }
    }

    pub(crate) fn root(&self) -> Option<&Node<(), T>> {
        self.root.map(|node| unsafe { node.as_ref() })
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn max_key(&self) -> usize {
        (1 << self.depth) - 1
    }

    pub fn insert_with<F>(&mut self, key: &[usize; 3], content: F) -> Option<T>
    where
        F: FnOnce() -> T,
    {
        let root = self.root.get_or_insert_with(|| {
            Box::leak(Box::new(Node::Branch {
                children: [None; 8],
                _content: (),
            }))
            .into()
        });
        unsafe { root.as_mut() }.insert_with(key, self.depth, content)
    }

    pub fn insert(&mut self, key: &[usize; 3], content: T) -> Option<T> {
        self.insert_with(key, || content)
    }

    pub fn get_or_insert_with<F>(&mut self, key: &[usize; 3], content: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        let root = self.root.get_or_insert_with(|| {
            Box::leak(Box::new(Node::Branch {
                children: [None; 8],
                _content: (),
            }))
            .into()
        });
        unsafe { root.as_mut() }.get_or_insert_with(key, self.depth, content)
    }

    pub fn get_or_insert(&mut self, key: &[usize; 3], content: T) -> &mut T {
        self.get_or_insert_with(key, || content)
    }

    pub fn get<'a>(&'a self, key: &[usize; 3]) -> Option<&'a T> {
        self.root.and_then(|root| {
            let node = unsafe { root.as_ref() }.find(key, self.depth);
            match unsafe { node.as_ref() } {
                Node::Leaf { content } => Some(content),
                Node::Branch { .. } => None,
            }
        })
    }

    pub fn get_mut<'a>(&'a mut self, key: &[usize; 3]) -> Option<&'a mut T> {
        self.root.and_then(|root| {
            let mut node = unsafe { root.as_ref() }.find(key, self.depth);
            match unsafe { node.as_mut() } {
                Node::Leaf { content } => Some(content),
                Node::Branch { .. } => None,
            }
        })
    }

    pub fn remove(&mut self, key: &[usize; 3]) -> Option<T> {
        self.root
            .and_then(|mut root| unsafe { root.as_mut() }.remove(key, self.depth))
    }
}

impl<T> OcTree<T> {
    pub fn encode(&self, mut output: impl io::Write) -> io::Result<Vec<T>>
    where
        T: Copy,
    {
        let mut leaves = Vec::new();
        if let Some(root) = self.root {
            unsafe { root.as_ref() }.encode(&mut output, &mut leaves)?;
        }
        Ok(leaves)
    }

    pub fn decode(
        mut input: impl io::Read,
        leaves: impl IntoIterator<Item = T>,
        depth: usize,
    ) -> io::Result<Self> {
        let depth_mask = 1 << (depth - 1);
        let root = Node::decode(&mut input, &mut leaves.into_iter(), depth_mask)?;
        Ok(OcTree {
            root: Some(root),
            depth,
        })
    }
}

impl<T> Drop for OcTree<T> {
    fn drop(&mut self) {
        if let Some(mut root) = self.root {
            unsafe {
                root.as_mut().destroy_subtree();
                let _ = Box::from_raw(root.as_ptr());
            }
        }
    }
}

impl<T> OcTree<T> {
    pub(crate) fn node_depth_iter(&self) -> NodeDepthIter<T> {
        NodeDepthIter {
            inner: self.root.map(crate::iter::RawDepthIter::new),
        }
    }

    pub(crate) fn node_depth_iter_mut(&mut self) -> NodeDepthIterMut<T> {
        NodeDepthIterMut {
            inner: self.root.map(crate::iter::RawDepthIter::new),
        }
    }

    pub fn depth_iter(&self) -> DepthIter<T> {
        DepthIter {
            inner: self.node_depth_iter(),
        }
    }

    pub fn depth_iter_mut(&mut self) -> DepthIterMut<T> {
        DepthIterMut {
            inner: self.node_depth_iter_mut(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oc_tree() {
        let mut tree = OcTree::new(3);
        assert_eq!(tree.max_key(), 7);

        let ret = tree.insert(&[3, 2, 3], 123);
        assert_eq!(ret, None);

        let ret = tree.get(&[3, 2, 3]);
        assert_eq!(ret, Some(&123));

        let ret = tree.insert(&[3, 2, 3], 234);
        assert_eq!(ret, Some(123));

        let ret = tree.get(&[3, 2, 3]);
        assert_eq!(ret, Some(&234));

        tree.insert(&[1, 4, 1], 34634);
        tree.insert(&[6, 2, 5], 23424);
        tree.insert(&[2, 5, 3], 34323);
        tree.insert(&[7, 1, 6], 64352);

        let ret = tree.remove(&[2, 5, 3]);
        assert_eq!(ret, Some(34323));

        let ret = tree.get(&[2, 5, 3]);
        assert_eq!(ret, None);
    }

    #[test]
    fn test_oc_tree_serde() {
        let mut tree = OcTree::new(2);
        assert_eq!(tree.max_key(), 3);

        tree.insert(&[0, 0, 0], 0);
        tree.insert(&[2, 3, 2], 232);

        let mut data = Vec::new();
        let ret = tree.encode(&mut data).unwrap();
        assert_eq!(ret, vec![0, 232]);
        assert_eq!(data, vec![0b_1000_0001, 0b_0000_0001, 0b_0000_0100]);

        let mut de = OcTree::decode(&*data, ret, 2).unwrap();
        let ret = de.get(&[0, 0, 0]);
        assert_eq!(ret, Some(&0));

        let ret = de.get_mut(&[2, 3, 2]);
        assert_eq!(ret, Some(&mut 232));
    }
}
