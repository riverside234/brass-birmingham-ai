//! Owner-local generational storage; typed semantic equality resolves hash collisions.
use super::{check, invalid, math::Statistics, Info, Result};
use slotmap::{new_key_type, SlotMap};
use std::{
    collections::{BTreeMap, HashMap},
    hash::Hash,
    sync::atomic::{AtomicU64, Ordering},
};

new_key_type! { struct NodeKey; }
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Handle {
    owner: u64,
    key: NodeKey,
}
pub struct Node<K, A> {
    pub info: Info<K, A>,
    pub stats: Statistics,
    pub ordinal: usize,
}
pub struct Tree<K, A> {
    owner: u64,
    nodes: SlotMap<NodeKey, Node<K, A>>,
    lookup: HashMap<std::sync::Arc<K>, NodeKey>,
    order: Vec<NodeKey>,
    max_nodes: usize,
    max_edges: usize,
    edges: usize,
    retention: Option<Retention>,
    retained_len: usize,
}
// One first-expansion parent per node gives a bounded, conservative descendant
// forest. Missing alternate links can lose a reuse hit, never admit a sibling.
#[derive(Default)]
struct Retention {
    parents: HashMap<NodeKey, (Handle, usize)>,
    undo: BTreeMap<(NodeKey, usize), (u64, [f64; 4])>,
    checkpoint_len: usize,
    watermark: u64,
}
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct RetainedCounts {
    pub original_nodes: usize,
    pub retained_nodes: usize,
    pub discarded_nodes: usize,
    pub retained_edges: usize,
    /// Edge contributions below the promoted root, not new trajectories.
    pub inherited_edge_visits: u64,
    pub reset_root_visits: u64,
    pub rolled_back_edge_updates: u64,
    pub watermark: u64,
}
#[derive(Debug, Clone)]
pub struct PathEdge {
    pub node: Handle,
    pub edge: usize,
    pub spent: [f64; 4],
}
pub(super) struct Backup {
    changes: BTreeMap<(NodeKey, usize), (u64, [f64; 4])>,
    values: Vec<[f64; 4]>,
}
impl<K: Eq + Hash + Clone, A: Eq + Ord + Clone> Tree<K, A> {
    pub fn new(max_nodes: usize, max_edges: usize) -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        check(max_nodes > 0 && max_edges > 0, "invalid arena limits")?;
        let owner = NEXT
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| invalid("arena identity exhausted"))?;
        Ok(Self {
            owner,
            nodes: SlotMap::with_key(),
            lookup: HashMap::new(),
            order: vec![],
            max_nodes,
            max_edges,
            edges: 0,
            retention: None,
            retained_len: 0,
        })
    }
    pub fn get(&self, h: Handle) -> Result<&Node<K, A>> {
        check(h.owner == self.owner, "foreign arena handle")?;
        self.nodes
            .get(h.key)
            .ok_or_else(|| invalid("stale arena handle"))
    }
    pub fn find(&self, info: &Info<K, A>) -> Result<Option<Handle>> {
        if let Some(key) = self.lookup.get(&info.key) {
            let h = Handle {
                owner: self.owner,
                key: *key,
            };
            check(
                self.get(h)?.info == *info,
                "information identity has inconsistent actor/legal semantics",
            )?;
            Ok(Some(h))
        } else {
            Ok(None)
        }
    }
    pub fn insert(&mut self, info: Info<K, A>, stats: Statistics) -> Result<Handle> {
        crate::work::checkpoint(crate::work::Stage::Expansion)?;
        info.validate()?;
        check(
            stats.count.len() == stats.logits.len()
                && stats.returns.len() == stats.logits.len()
                && stats.prior.len() == stats.logits.len()
                && stats.count.iter().all(|n| *n == 0)
                && stats.returns.iter().all(|r| r.iter().all(|x| *x == 0.))
                && stats
                    .logits
                    .iter()
                    .chain(&stats.value)
                    .chain(&stats.prior)
                    .all(|x| x.is_finite()),
            "invalid fresh node statistics",
        )?;
        check(self.find(&info)?.is_none(), "duplicate node expansion")?;
        check(
            stats.logits.len() == info.actions.len(),
            "leaf legal width mismatch",
        )?;
        check(
            self.nodes.len() < self.max_nodes
                && info.actions.len() <= self.max_edges.saturating_sub(self.edges),
            "tree memory cap",
        )?;
        let lookup = info.key.clone();
        self.edges += info.actions.len();
        let key = self.nodes.insert(Node {
            info,
            stats,
            ordinal: self.order.len(),
        });
        self.order.push(key);
        self.lookup.insert(lookup, key);
        Ok(Handle {
            owner: self.owner,
            key,
        })
    }
    pub fn ordered(&self) -> impl Iterator<Item = &Node<K, A>> {
        self.order.iter().filter_map(|k| self.nodes.get(*k))
    }
    /// Opt-in only; the baseline does not maintain links or a round journal.
    pub fn enable_retention(&mut self) {
        self.retention = Some(Retention {
            checkpoint_len: self.order.len(),
            ..Default::default()
        });
    }
    pub fn insert_child(
        &mut self,
        info: Info<K, A>,
        stats: Statistics,
        parent: &PathEdge,
    ) -> Result<Handle> {
        if self.retention.is_some() {
            check(
                parent.edge < self.get(parent.node)?.info.actions.len(),
                "invalid parent edge",
            )?;
        }
        let h = self.insert(info, stats)?;
        if let Some(r) = &mut self.retention {
            r.parents.insert(h.key, (parent.node, parent.edge));
        }
        Ok(h)
    }
    /// Called immediately after publication of an eligible completed round.
    pub fn checkpoint_round(&mut self, watermark: u64) {
        if let Some(r) = &mut self.retention {
            r.undo.clear();
            r.checkpoint_len = self.order.len();
            r.watermark = watermark;
        }
    }
    fn journal(&mut self, changes: &BTreeMap<(NodeKey, usize), (u64, [f64; 4])>) {
        if let Some(r) = &mut self.retention {
            for &(key, edge) in changes.keys() {
                // Newly expanded nodes are removed wholesale on rollback.
                let n = &self.nodes[key];
                if n.ordinal < r.checkpoint_len {
                    r.undo
                        .entry((key, edge))
                        .or_insert((n.stats.count[edge], n.stats.returns[edge]));
                }
            }
        }
    }
    /// Move this owner-local arena to a certified descendant. Verification and
    /// rollback/pruning are bounded by existing arena caps and poll the caller's
    /// original clock. On error the caller discards this arena; no live state or
    /// original tournament is resumed after an accepted switch.
    pub fn promote(
        &mut self,
        old_root: Handle,
        prefix: &[(Info<K, A>, A)],
        boundary: &Info<K, A>,
        watermark: u64,
        check_time: &mut impl FnMut() -> Result<()>,
    ) -> Result<(Handle, RetainedCounts)> {
        self.get(old_root)?;
        check_time()?;
        let r = self
            .retention
            .as_ref()
            .ok_or_else(|| invalid("retention disabled"))?;
        check(
            r.watermark == watermark && watermark > 0,
            "retention watermark mismatch",
        )?;
        let root = self
            .find(boundary)?
            .ok_or_else(|| invalid("no matching descendant"))?;
        check(
            self.get(root)?.ordinal < r.checkpoint_len,
            "descendant exists only after watermark",
        )?;
        let mut cursor = root;
        for (info, action) in prefix.iter().rev() {
            check_time()?;
            let (parent, edge) = *r
                .parents
                .get(&cursor.key)
                .ok_or_else(|| invalid("prefix parent missing"))?;
            let n = self.get(parent)?;
            check(
                n.info == *info && n.info.actions.get(edge) == Some(action),
                "prefix parent mismatch",
            )?;
            cursor = parent;
        }
        check(
            !prefix.is_empty() && cursor == old_root,
            "prefix does not reach original root",
        )?;
        let mut r = self.retention.take().expect("checked retention");
        let mut counts = RetainedCounts {
            original_nodes: self.len(),
            watermark,
            ..Default::default()
        };
        for ((key, edge), (count, returns)) in std::mem::take(&mut r.undo) {
            check_time()?;
            let n = &mut self.nodes[key];
            counts.rolled_back_edge_updates += n.stats.count[edge] - count;
            n.stats.count[edge] = count;
            n.stats.returns[edge] = returns;
        }
        // Parents always precede their first expanded child. A single ordered
        // pass finds the subtree without a graph scan or copying node contents.
        let mut retained = std::collections::HashSet::new();
        for (index, key) in self.order.iter().copied().enumerate() {
            check_time()?;
            if index < r.checkpoint_len
                && (key == root.key
                    || r.parents
                        .get(&key)
                        .is_some_and(|(p, _)| retained.contains(&p.key)))
            {
                retained.insert(key);
            }
        }
        let old_order = std::mem::take(&mut self.order);
        self.edges = 0;
        for key in old_order {
            check_time()?;
            if retained.contains(&key) {
                let n = &mut self.nodes[key];
                n.ordinal = self.order.len();
                self.order.push(key);
                self.edges += n.info.actions.len();
                let visits = n.stats.count.iter().sum::<u64>();
                if key == root.key {
                    counts.reset_root_visits = visits;
                    n.stats.count.fill(0);
                    n.stats.returns.fill([0.; 4]);
                } else {
                    counts.inherited_edge_visits += visits;
                }
            } else {
                let node = self.nodes.remove(key).expect("owned node");
                self.lookup.remove(&node.info.key);
            }
        }
        counts.retained_nodes = self.len();
        counts.retained_edges = self.edges;
        counts.discarded_nodes = counts.original_nodes - counts.retained_nodes;
        self.retained_len = self.len();
        // Only one switch is allowed: no journal/links are needed in phase two.
        check_time()?;
        Ok((root, counts))
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    pub fn edges(&self) -> usize {
        self.edges
    }
    pub fn retained_len(&self) -> usize {
        self.retained_len
    }
    /// Preflight the whole bounded path, including repeated edges, before publication.
    /// No whole-tree copy and no locks. Costs are charged only after each edge's view.
    pub fn backup(
        &mut self,
        path: &[PathEdge],
        value: [f64; 4],
        spent: [f64; 4],
    ) -> Result<Vec<[f64; 4]>> {
        let staged = self.prepare_backup(path, value, spent)?;
        Ok(self.publish_backup(staged))
    }
    pub(super) fn prepare_backup(
        &self,
        path: &[PathEdge],
        value: [f64; 4],
        spent: [f64; 4],
    ) -> Result<Backup> {
        check(
            value.iter().chain(&spent).all(|x| x.is_finite()) && spent.iter().all(|x| *x >= 0.),
            "invalid leaf return/cost",
        )?;
        let mut changes = BTreeMap::new();
        let mut values = Vec::with_capacity(path.len());
        for edge in path {
            crate::work::checkpoint(crate::work::Stage::Backup)?;
            let node = self.get(edge.node)?;
            check(edge.edge < node.stats.count.len(), "stale edge")?;
            check(
                edge.spent
                    .iter()
                    .zip(spent)
                    .all(|(a, b)| a.is_finite() && *a >= 0. && *a <= b),
                "cost ledger moved backwards",
            )?;
            let next = changes
                .entry((edge.node.key, edge.edge))
                .or_insert((node.stats.count[edge.edge], node.stats.returns[edge.edge]));
            next.0 = next
                .0
                .checked_add(1)
                .ok_or_else(|| invalid("visit overflow"))?;
            let v = std::array::from_fn(|s| value[s] - (spent[s] - edge.spent[s]));
            for (sum, x) in next.1.iter_mut().zip(v) {
                *sum += x;
            }
            check(next.1.iter().all(|x| x.is_finite()), "return overflow")?;
            values.push(v);
        }
        Ok(Backup { changes, values })
    }
    pub(super) fn publish_backup(&mut self, staged: Backup) -> Vec<[f64; 4]> {
        self.journal(&staged.changes);
        for ((key, edge), (count, returns)) in staged.changes {
            let node = &mut self.nodes[key];
            node.stats.count[edge] = count;
            node.stats.returns[edge] = returns;
        }
        staged.values
    }
    pub(super) fn commit_checked(
        &mut self,
        staged: Backup,
        inserted: Option<Handle>,
        check_time: &mut impl FnMut() -> Result<()>,
    ) -> Result<()> {
        self.journal(&staged.changes);
        let undo: Vec<_> = staged
            .changes
            .keys()
            .map(|(key, edge)| {
                let n = &self.nodes[*key];
                (*key, *edge, n.stats.count[*edge], n.stats.returns[*edge])
            })
            .collect();
        let result = (|| {
            for ((key, edge), (count, returns)) in staged.changes {
                crate::work::checkpoint(crate::work::Stage::Publication)?;
                check_time()?;
                let node = &mut self.nodes[key];
                node.stats.count[edge] = count;
                node.stats.returns[edge] = returns;
            }
            crate::work::checkpoint(crate::work::Stage::Publication)?;
            check_time()
        })();
        if result.is_err() {
            for (key, edge, count, returns) in undo {
                let node = &mut self.nodes[key];
                node.stats.count[edge] = count;
                node.stats.returns[edge] = returns;
            }
            if let Some(h) = inserted {
                let node = self.nodes.remove(h.key).expect("unpublished expansion");
                self.lookup.remove(&node.info.key);
                self.order.pop();
                self.edges -= node.info.actions.len();
                if let Some(r) = &mut self.retention {
                    r.parents.remove(&h.key);
                }
            }
        }
        result
    }
    /// Root changes discard counts. Slot generations invalidate outstanding handles.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.lookup.clear();
        self.order.clear();
        self.edges = 0;
        self.retention = None;
        self.retained_len = 0;
    }
}
