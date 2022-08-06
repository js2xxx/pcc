use std::{mem, ptr::NonNull};

pub(crate) enum Node<B, L> {
    Leaf {
        content: L,
    },
    Branch {
        children: [Option<NonNull<Node<B, L>>>; 8],
        _content: B,
    },
}

fn key_to_index(key: &[usize; 3], depth_mask: usize) -> usize {
    (((key[2] & depth_mask != 0) as usize) << 2)
        | (((key[1] & depth_mask != 0) as usize) << 1)
        | (key[0] & depth_mask != 0) as usize
}

impl<B, L> Node<B, L> {
    
    pub fn find(&self, key: &[usize; 3], depth: usize) -> Option<NonNull<L>> {
        let mut node = self;
        let mut depth_mask = (1 << depth) - 1;
        loop {
            match node {
                Node::Leaf { content } => break Some(content.into()),
                Node::Branch { children, .. } => {
                    let index = key_to_index(key, depth_mask);
                    let child = match children[index] {
                        Some(child) => child,
                        None => break None,
                    };
                    node = unsafe { child.as_ref() };
                    depth_mask >>= 1;
                }
            }
        }
    }

    pub fn insert_with<F>(&mut self, key: &[usize; 3], depth: usize, content: F) -> Option<L>
    where
        F: FnOnce() -> L,
        B: Default,
    {
        let mut node = self;
        let mut depth_mask = (1 << depth) - 1;
        loop {
            match node {
                Node::Leaf { content: c } => break Some(mem::replace(c, content())),
                Node::Branch { children, .. } => {
                    let index = key_to_index(key, depth_mask);

                    let child = match &mut children[index] {
                        Some(child) => child,
                        child @ None if depth_mask > 1 => {
                            let data = Node::Branch {
                                children: [None; 8],
                                _content: Default::default(),
                            };
                            child.insert(Box::leak(Box::new(data)).into())
                        }
                        child @ None => {
                            let data = Node::Leaf { content: content() };
                            *child = Some(Box::leak(Box::new(data)).into());

                            break None;
                        }
                    };

                    node = unsafe { child.as_mut() };
                    depth_mask >>= 1;
                }
            }
        }
    }

    pub fn remove(&mut self, key: &[usize; 3], depth: usize) -> Option<L> {
        match self {
            Node::Leaf { .. } => panic!("Leaf node with no parents can't be removed here"),
            Node::Branch { children, .. } => remove_recursive(children, key, (1 << depth) - 1),
        }
    }
}

fn remove_recursive<B, L>(
    children: &mut [Option<NonNull<Node<B, L>>>; 8],
    key: &[usize; 3],
    depth_mask: usize,
) -> Option<L> {
    let index = key_to_index(key, depth_mask);
    let mut child = match children[index] {
        Some(child) => child,
        None => return None,
    };

    let child = unsafe { child.as_mut() };
    match child {
        Node::Leaf { .. } => {
            let data = children[index].take().unwrap();
            let data = unsafe { Box::into_inner(Box::from_raw(data.as_ptr())) };
            match data {
                Node::Leaf { content } => Some(content),
                _ => unreachable!(),
            }
        }
        Node::Branch { children: cc, .. } => {
            let ret = remove_recursive(cc, key, depth_mask >> 1);

            if ret.is_some() && cc.iter().all(|child| child.is_none()) {
                let data = children[index].take().unwrap();
                let _ = unsafe { Box::from_raw(data.as_ptr()) };
            }

            ret
        }
    }
}
