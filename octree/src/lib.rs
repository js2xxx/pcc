#![feature(array_try_from_fn)]
#![feature(box_into_inner)]

use std::{io, ptr::NonNull};

use node::Node;

mod node;

pub struct OcTree<T> {
    root: Option<NonNull<Node<(), T>>>,
    depth: usize,
}

impl<T> OcTree<T> {
    pub fn new(depth: usize) -> Self {
        OcTree { root: None, depth }
    }

    pub fn depth(&self) -> usize {
        self.depth
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

    pub fn get<'a>(&'a self, key: &[usize; 3]) -> Option<&'a T> {
        self.root
            .and_then(|root| unsafe { root.as_ref() }.find(key, self.depth))
            .map(|content| unsafe { content.as_ref() })
    }

    pub fn get_mut<'a>(&'a mut self, key: &[usize; 3]) -> Option<&'a mut T> {
        self.root
            .and_then(|root| unsafe { root.as_ref() }.find(key, self.depth))
            .map(|mut content| unsafe { content.as_mut() })
    }

    pub fn remove(&mut self, key: &[usize; 3]) -> Option<T> {
        self.root
            .and_then(|mut root| unsafe { root.as_mut() }.remove(key, self.depth))
    }
}

impl<T> OcTree<T> {
    pub fn encode(&self, output: &mut impl io::Write) -> io::Result<Vec<T>>
    where
        T: Copy,
    {
        let mut leaves = Vec::new();
        if let Some(root) = self.root {
            unsafe { root.as_ref() }.encode(output, &mut leaves)?;
        }
        Ok(leaves)
    }

    pub fn decode(
        input: &mut impl io::Read,
        leaves: &mut impl Iterator<Item = T>,
        depth: usize,
    ) -> io::Result<Self> {
        let depth_mask = (1 << depth) - 1;
        let root = Node::decode(input, leaves, depth_mask)?;
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
