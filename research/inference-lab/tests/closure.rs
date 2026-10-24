use inference_lab::{Evidence, Program, Rule, delta_rounds, naive, relational, worklist};
use std::collections::{HashMap, HashSet};

fn verify(seeds: &[u32], rules: &[Rule]) -> HashSet<u32> {
    let reference = naive(seeds, rules).facts;
    assert_eq!(delta_rounds(seeds, rules).facts, reference);
    let program = Program::new(rules);
    let order: Vec<_> = program.derive(seeds.iter().copied(), true).collect();
    assert_eq!(order.iter().copied().collect::<HashSet<_>>(), reference);
    assert_eq!(order.len(), reference.len());
    let positions: HashMap<_, _> = order.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let o = worklist(seeds, rules, true);
    for (&f, e) in o.evidence.as_ref().unwrap() {
        match e {
            Evidence::Seed => assert!(seeds.contains(&f)),
            Evidence::Rule(id) => {
                assert_eq!(rules[*id].conclusion, f);
                for p in &rules[*id].premises {
                    assert!(positions[p] < positions[&f]);
                }
            }
        }
    }
    reference
}

#[test]
fn edge_cases_and_proofs() {
    let rules = [
        Rule::new(vec![0, 0, 1], 2),
        Rule::new(vec![2], 0),
        Rule::new(vec![], 1),
        Rule::new(vec![2], 3),
        Rule::new(vec![9], 10),
        Rule::new(vec![], 1),
        Rule::new(vec![7], 7),
    ];
    assert_eq!(verify(&[0, 0], &rules), HashSet::from([0, 1, 2, 3]));
    assert!(verify(&[], &[]).is_empty());
    assert_eq!(verify(&[42], &[]), HashSet::from([42]));
}

#[test]
fn closure_axioms_and_reference_agreement_generated() {
    let mut state = 0x123456789abcdefu64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _case in 0..160 {
        let rules: Vec<_> = (0..48)
            .map(|_| {
                let count = (next() % 5) as usize;
                let premises = (0..count).map(|_| (next() % 24) as u32).collect::<Vec<_>>();
                Rule::new(premises, (next() % 24) as u32)
            })
            .collect();
        let a: Vec<_> = (0..5).map(|_| (next() % 24) as u32).collect();
        let mut b = a.clone();
        b.extend((0..5).map(|_| (next() % 24) as u32));
        let ca = verify(&a, &rules);
        let cb = verify(&b, &rules);
        assert!(a.iter().all(|f| ca.contains(f)), "extensivity");
        assert!(ca.is_subset(&cb), "monotonicity");
        let closure_seeds: Vec<_> = ca.iter().copied().collect();
        assert_eq!(
            worklist(&closure_seeds, &rules, false).facts,
            ca,
            "idempotence"
        );
    }
}

#[test]
fn initialization_eager_iteration_lazy_and_fused() {
    use std::cell::Cell;
    let consumed = Cell::new(0);
    let rules = [Rule::new(vec![0], 1), Rule::new(vec![1], 2)];
    let program = Program::new(&rules);
    let mut iter = program.derive(
        [0].into_iter()
            .inspect(|_| consumed.set(consumed.get() + 1)),
        false,
    );
    assert_eq!(consumed.get(), 1);
    assert_eq!(iter.stats().dependency_events, 0);
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.stats().dependency_events, 1);
    // Dropping here must not run the rule depending on 1.
    drop(iter);
    let mut iter = program.derive([0], false);
    assert_eq!(iter.by_ref().collect::<Vec<_>>(), vec![0, 1, 2]);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
}

#[test]
fn snapshot_recomputation_removes_stale_conclusions() {
    let rules = [Rule::new(vec![0], 1), Rule::new(vec![1], 2)];
    assert_eq!(worklist(&[0], &rules, false).facts.len(), 3);
    assert!(worklist(&[], &rules, false).facts.is_empty());
}

#[test]
fn relational_join_matches_independent_graph_search() {
    let edges = [(0, 1), (1, 2), (2, 0), (2, 3), (0, 1), (9, 10)];
    let mut expected = HashSet::new();
    for source in 0..=10 {
        let mut frontier = vec![source];
        let mut visited = HashSet::new();
        while let Some(x) = frontier.pop() {
            for &(a, b) in &edges {
                if a == x && visited.insert(b) {
                    expected.insert((source, b));
                    frontier.push(b);
                }
            }
        }
    }
    assert_eq!(relational::reachability(&edges, false).paths, expected);
    assert_eq!(relational::reachability(&edges, true).paths, expected);
}
