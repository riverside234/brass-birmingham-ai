use super::{catalog::CATALOG, dto::Industry, seat, Seat, TileId, TileState};
use brass_upstream_rules::utils::dsu::DisjointSetUnion;
use fixedbitset::FixedBitSet;
use serde::Serialize;

/// Physical indexes only. Resource payment legality and distances belong to L05.
#[derive(Debug, Clone)]
pub(super) struct Indexes {
    connectivity: DisjointSetUnion,
    owned_slots: [FixedBitSet; 4],
    owned_links: [FixedBitSet; 4],
    network_regions: [FixedBitSet; 4],
    resources: [FixedBitSet; 3],
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndexSnapshot {
    pub connected_regions: Vec<Vec<String>>,
    pub owned_slots: [Vec<String>; 4],
    pub owned_links: [Vec<String>; 4],
    pub network_regions: [Vec<String>; 4],
    pub coal_slots: Vec<String>,
    pub iron_slots: Vec<String>,
    pub beer_slots: Vec<String>,
}
impl Indexes {
    pub fn connected(&self, a: usize, b: usize) -> bool {
        self.connectivity.find_set_immutable(a) == self.connectivity.find_set_immutable(b)
    }
    pub fn in_network(&self, player: super::Seat, region: usize) -> bool {
        let network = &self.network_regions[seat(player)];
        network.is_clear() || network.contains(region)
    }
    pub fn empty() -> Self {
        Self {
            connectivity: DisjointSetUnion::new(CATALOG.regions.len()),
            owned_slots: std::array::from_fn(|_| FixedBitSet::with_capacity(CATALOG.slots.len())),
            owned_links: std::array::from_fn(|_| FixedBitSet::with_capacity(CATALOG.links.len())),
            network_regions: std::array::from_fn(|_| {
                FixedBitSet::with_capacity(CATALOG.regions.len())
            }),
            resources: std::array::from_fn(|_| FixedBitSet::with_capacity(CATALOG.slots.len())),
        }
    }
    /// Fresh indexes for `s`: the reference that `refill` is tested against.
    #[cfg(test)]
    pub fn build(s: &super::NativeState) -> Self {
        let mut indexes = Self::empty();
        indexes.fill(&s.links, &s.slots, &s.tiles);
        indexes
    }
    /// `build` for this board, reusing these indexes' storage. A reset union-find
    /// and cleared sets are exactly `empty()`, so the result equals `build`.
    pub fn refill(&mut self, links: &[Option<Seat>], slots: &[Option<TileId>], tiles: &[TileState]) {
        self.connectivity.reset();
        for set in self
            .owned_slots
            .iter_mut()
            .chain(self.owned_links.iter_mut())
            .chain(self.network_regions.iter_mut())
            .chain(self.resources.iter_mut())
        {
            set.clear();
        }
        self.fill(links, slots, tiles);
    }
    fn fill(&mut self, links: &[Option<Seat>], slots: &[Option<TileId>], tiles: &[TileState]) {
        for (i, owner) in links.iter().enumerate() {
            if let Some(owner) = owner {
                self.owned_links[seat(*owner)].insert(i);
                let regions = &CATALOG.link_regions[i];
                for r in regions {
                    self.connectivity.union_sets(regions[0], *r);
                    self.network_regions[seat(*owner)].insert(*r);
                }
            }
        }
        for (i, tile) in slots.iter().enumerate() {
            if let Some(id) = tile {
                let def = &CATALOG.tiles[id.0];
                self.owned_slots[seat(def.seat)].insert(i);
                self.network_regions[seat(def.seat)].insert(CATALOG.slot_regions[i]);
                if tiles[id.0].resource > 0 {
                    let resource = match def.industry {
                        Industry::CoalMine => Some(0),
                        Industry::IronWorks => Some(1),
                        Industry::Brewery => Some(2),
                        _ => None,
                    };
                    if let Some(resource) = resource {
                        self.resources[resource].insert(i);
                    }
                }
            }
        }
    }
    pub fn export(&self) -> IndexSnapshot {
        let mut connected_regions = Vec::new();
        let mut seen = FixedBitSet::with_capacity(CATALOG.regions.len());
        for i in 0..CATALOG.regions.len() {
            if !seen.contains(i) {
                let group = self.connectivity.get_set_elements_immutable(i);
                for r in &group {
                    seen.insert(*r);
                }
                connected_regions.push(group.iter().map(|r| CATALOG.regions[*r].clone()).collect());
            }
        }
        IndexSnapshot {
            connected_regions,
            owned_slots: std::array::from_fn(|i| {
                self.owned_slots[i]
                    .ones()
                    .map(|id| CATALOG.slots[id].id.to_string())
                    .collect()
            }),
            owned_links: std::array::from_fn(|i| {
                self.owned_links[i]
                    .ones()
                    .map(|id| CATALOG.links[id].id.to_string())
                    .collect()
            }),
            network_regions: std::array::from_fn(|i| {
                self.network_regions[i]
                    .ones()
                    .map(|id| CATALOG.regions[id].clone())
                    .collect()
            }),
            coal_slots: self.resources[0]
                .ones()
                .map(|id| CATALOG.slots[id].id.to_string())
                .collect(),
            iron_slots: self.resources[1]
                .ones()
                .map(|id| CATALOG.slots[id].id.to_string())
                .collect(),
            beer_slots: self.resources[2]
                .ones()
                .map(|id| CATALOG.slots[id].id.to_string())
                .collect(),
        }
    }
}
