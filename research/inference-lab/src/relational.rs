//! A deliberately narrow relational experiment:
//! path(x,y) <- edge(x,y); path(x,z) <- path(x,y), edge(y,z).
//! Both evaluators compute positive-length reachability with the same join index.

use std::collections::{HashMap, HashSet};

pub type Pair = (u32, u32);

pub struct Reachability {
    pub paths: HashSet<Pair>,
    pub rounds: usize,
    pub join_probes: usize,
    pub join_matches: usize,
}

pub fn reachability(edges: &[Pair], semi_naive: bool) -> Reachability {
    let mut by_source: HashMap<u32, HashSet<u32>> = HashMap::new();
    for &(a, b) in edges {
        by_source.entry(a).or_default().insert(b);
    }
    let mut paths: HashSet<_> = edges.iter().copied().collect();
    let mut delta = paths.clone();
    let mut rounds = 0;
    let mut join_probes = 0;
    let mut join_matches = 0;
    loop {
        rounds += 1;
        let input = if semi_naive { &delta } else { &paths };
        let mut next = HashSet::new();
        for &(x, y) in input {
            join_probes += 1;
            if let Some(targets) = by_source.get(&y) {
                for &z in targets {
                    join_matches += 1;
                    if !paths.contains(&(x, z)) {
                        next.insert((x, z));
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        paths.extend(next.iter().copied());
        delta = next;
    }
    Reachability {
        paths,
        rounds,
        join_probes,
        join_matches,
    }
}
