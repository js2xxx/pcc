#![feature(box_into_inner)]

use node::Node;

mod node;

pub struct OcTree<T> {
    root: Node<(), T>,
    depth: usize,
}

impl<T> OcTree<T> {
    pub fn new(depth: usize) -> Self {
        OcTree {
            root: Node::Branch {
                children: [None; 8],
                _content: (),
            },
            depth,
        }
    }

    pub fn insert_with<F>(&mut self, key: &[usize; 3], content: F) -> Option<T>
    where
        F: FnOnce() -> T,
    {
        self.root.insert_with(key, self.depth, content)
    }

    pub fn insert(&mut self, key: &[usize; 3], content: T) -> Option<T> {
        self.insert_with(key, || content)
    }

    pub fn get<'a>(&'a self, key: &[usize; 3]) -> Option<&'a T> {
        self.root
            .find(key, self.depth)
            .map(|content| unsafe { content.as_ref() })
    }

    pub fn get_mut<'a>(&'a mut self, key: &[usize; 3]) -> Option<&'a mut T> {
        self.root
            .find(key, self.depth)
            .map(|mut content| unsafe { content.as_mut() })
    }

    pub fn remove(&mut self, key: &[usize; 3]) -> Option<T> {
        self.root.remove(key, self.depth)
    }
}
