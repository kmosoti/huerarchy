//! A Rust FnMut closure adapts the reusable inference iterator.
//! The closure is an execution mechanism; mathematical closure is its result.
use inference_lab::{Program, Rule};
use std::collections::HashSet;

fn main() {
    let names = ["A", "B", "C", "D"];
    let rules = [
        Rule::new(vec![0, 1], 2),
        Rule::new(vec![2], 3),
        Rule::new(vec![3], 0),
    ];
    let program = Program::new(&rules);
    let mut derivation = program.derive([0, 1], false);
    let lazy = std::iter::from_fn(move || derivation.next());
    let facts: Vec<_> = lazy.map(|fact| names[fact as usize]).collect();
    assert_eq!(
        facts.iter().copied().collect::<HashSet<_>>(),
        HashSet::from(names)
    );
    println!("{}", facts.join(", "));
}
