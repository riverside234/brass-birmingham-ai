use super::{catalog::CATALOG, dto::Industry, seat, NativeState};
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
    pub fn build(s: &NativeState) -> Self {
        let mut indexes = Self::empty();
        for (i, owner) in s.links.iter().enumerate() {
            if let Some(owner) = owner {
                indexes.owned_links[seat(*owner)].insert(i);
                let regions = &CATALOG.link_regions[i];
                for r in regions {
                    indexes.connectivity.union_sets(regions[0], *r);
                    indexes.network_regions[seat(*owner)].insert(*r);
                }
            }
        }
        for (i, tile) in s.slots.iter().enumerate() {
            if let Some(id) = tile {
                let def = &CATALOG.tiles[id.0];
                indexes.owned_slots[seat(def.seat)].insert(i);
                indexes.network_regions[seat(def.seat)].insert(CATALOG.slot_regions[i]);
                if s.tiles[id.0].resource > 0 {
                    let resource = match def.industry {
                        Industry::CoalMine => Some(0),
                        Industry::IronWorks => Some(1),
                        Industry::Brewery => Some(2),
                        _ => None,
                    };
                    if let Some(resource) = resource {
                        indexes.resources[resource].insert(i);
                    }
                }
            }
        }
        indexes
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
                    .map(|id| CATALOG.slots[id].id.clone())
                    .collect()
            }),
            owned_links: std::array::from_fn(|i| {
                self.owned_links[i]
                    .ones()
                    .map(|id| CATALOG.links[id].id.clone())
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
                .map(|id| CATALOG.slots[id].id.clone())
                .collect(),
            iron_slots: self.resources[1]
                .ones()
                .map(|id| CATALOG.slots[id].id.clone())
                .collect(),
            beer_slots: self.resources[2]
                .ones()
                .map(|id| CATALOG.slots[id].id.clone())
                .collect(),
        }
    }
}
