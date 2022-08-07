use std::{marker::PhantomData, ptr::NonNull};

use crate::node::Node;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct DepthIterItem<'a, B, L> {
    node: NonNull<Node<B, L>>,
    index: usize,
    _marker: PhantomData<&'a Node<B, L>>,
}

impl<'a, B, L> DepthIterItem<'a, B, L> {
    fn new(node: NonNull<Node<B, L>>, index: usize) -> Self {
        DepthIterItem {
            node,
            index,
            _marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawDepthIter<'a, B, L> {
    stack: Vec<DepthIterItem<'a, B, L>>,
}

impl<'a, B, L> RawDepthIter<'a, B, L> {
    pub fn new(node: NonNull<Node<B, L>>) -> Self {
        RawDepthIter {
            stack: vec![DepthIterItem::new(node, 0)],
        }
    }
}

impl<'a, B, L> Iterator for RawDepthIter<'a, B, L> {
    type Item = NonNull<Node<B, L>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let back = self.stack.last_mut()?;

            let children = match unsafe { back.node.as_ref() } {
                Node::Leaf { .. } => panic!("Leaf node without parents can't be directly iterated"),
                Node::Branch { children, .. } => children,
            };

            let ret = loop {
                if back.index >= 8 {
                    break None;
                }

                match children[back.index] {
                    None => back.index += 1,
                    Some(child) => {
                        back.index += 1;
                        break Some(child);
                    }
                }
            };

            match ret {
                Some(child) => {
                    let c = unsafe { child.as_ref() };
                    if matches!(c, Node::Branch { .. }) {
                        self.stack.push(DepthIterItem::new(child, 0))
                    }
                    break Some(child);
                }
                None => {
                    self.stack.pop();
                }
            }
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct NodeDepthIter<'a, T> {
    pub(crate) inner: Option<RawDepthIter<'a, (), T>>,
}

impl<'a, T> Iterator for NodeDepthIter<'a, T> {
    type Item = &'a Node<(), T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .as_mut()
            .and_then(|iter| iter.next().map(|node| unsafe { node.as_ref() }))
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct NodeDepthIterMut<'a, T> {
    pub(crate) inner: Option<RawDepthIter<'a, (), T>>,
}

impl<'a, T> Iterator for NodeDepthIterMut<'a, T> {
    type Item = &'a mut Node<(), T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .as_mut()
            .and_then(|iter| iter.next().map(|mut node| unsafe { node.as_mut() }))
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DepthIter<'a, T> {
    pub(crate) inner: NodeDepthIter<'a, T>,
}

impl<'a, T> Iterator for DepthIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next() {
                Some(node) => match node {
                    Node::Leaf { content } => break Some(content),
                    Node::Branch { .. } => {}
                },
                None => break None,
            }
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct DepthIterMut<'a, T> {
    pub(crate) inner: NodeDepthIterMut<'a, T>,
}

impl<'a, T> Iterator for DepthIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next() {
                Some(node) => match node {
                    Node::Leaf { content } => break Some(content),
                    Node::Branch { .. } => {}
                },
                None => break None,
            }
        }
    }
}
