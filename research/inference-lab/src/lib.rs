//! Experiments with finite, ground, positive Horn rules.
//! Facts are interned IDs. There is no negation, unification, or retraction.

use std::collections::{HashMap, HashSet, VecDeque};
use std::iter::FusedIterator;

pub type Fact = u32;

#[derive(Clone, Debug)]
pub struct Rule {
    pub premises: Vec<Fact>,
    pub conclusion: Fact,
}

impl Rule {
    pub fn new(premises: impl Into<Vec<Fact>>, conclusion: Fact) -> Self {
        Self {
            premises: premises.into(),
            conclusion,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub rounds: usize,
    pub rule_checks: usize,
    pub premise_checks: usize,
    pub dependency_events: usize,
    pub rule_firings: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence {
    Seed,
    /// Rule ID; its premises supply the incoming proof edges.
    Rule(usize),
}

#[derive(Debug)]
pub struct Outcome {
    pub facts: HashSet<Fact>,
    pub stats: Stats,
    /// One first-discovery proof per fact, not every possible proof.
    pub evidence: Option<HashMap<Fact, Evidence>>,
}

/// Synchronous rounds make this a simple independent reference implementation.
pub fn naive(seeds: &[Fact], rules: &[Rule]) -> Outcome {
    let mut facts: HashSet<_> = seeds.iter().copied().collect();
    let mut stats = Stats::default();
    loop {
        stats.rounds += 1;
        let mut delta = HashSet::new();
        for rule in rules {
            stats.rule_checks += 1;
            let satisfied = rule.premises.iter().all(|p| {
                stats.premise_checks += 1;
                facts.contains(p)
            });
            if satisfied {
                stats.rule_firings += 1;
                if !facts.contains(&rule.conclusion) {
                    delta.insert(rule.conclusion);
                }
            }
        }
        if delta.is_empty() {
            break;
        }
        facts.extend(delta);
    }
    Outcome {
        facts,
        stats,
        evidence: None,
    }
}

/// Delta rounds for *ground* rules. This is not a relational join engine.
/// Reconsider only rules with at least one premise in the previous delta.
pub fn delta_rounds(seeds: &[Fact], rules: &[Rule]) -> Outcome {
    let program = Program::new(rules);
    let mut facts: HashSet<_> = seeds.iter().copied().collect();
    let mut delta = facts.clone();
    let mut unconditional = Vec::new();
    let mut stats = Stats::default();
    for (id, rule) in rules.iter().enumerate() {
        if rule.premises.is_empty() {
            unconditional.push(id);
        }
    }
    loop {
        stats.rounds += 1;
        let mut candidates: HashSet<_> = unconditional.drain(..).collect();
        for fact in delta {
            if let Some(ids) = program.dependents.get(&fact) {
                stats.dependency_events += ids.len();
                candidates.extend(ids.iter().copied());
            }
        }
        let mut next = HashSet::new();
        for id in candidates {
            stats.rule_checks += 1;
            let rule = &rules[id];
            if rule.premises.iter().all(|p| {
                stats.premise_checks += 1;
                facts.contains(p)
            }) {
                stats.rule_firings += 1;
                if !facts.contains(&rule.conclusion) {
                    next.insert(rule.conclusion);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        facts.extend(next.iter().copied());
        delta = next;
    }
    Outcome {
        facts,
        stats,
        evidence: None,
    }
}

/// Compile once and reuse this immutable index across seed sets.
pub struct Program<'r> {
    rules: &'r [Rule],
    dependents: HashMap<Fact, Vec<usize>>,
}

impl<'r> Program<'r> {
    pub fn new(rules: &'r [Rule]) -> Self {
        let mut dependents: HashMap<Fact, Vec<usize>> = HashMap::new();
        for (id, rule) in rules.iter().enumerate() {
            for &premise in &rule.premises {
                // Count occurrences consistently with pending, including duplicates.
                dependents.entry(premise).or_default().push(id);
            }
        }
        Self { rules, dependents }
    }

    pub fn derive(
        &self,
        seeds: impl IntoIterator<Item = Fact>,
        proofs: bool,
    ) -> Derivation<'_, 'r> {
        let mut engine = Derivation {
            program: self,
            pending: self.rules.iter().map(|r| r.premises.len()).collect(),
            seen: HashSet::new(),
            queue: VecDeque::new(),
            stats: Stats::default(),
            evidence: proofs.then(HashMap::new),
        };
        // Initialization consumes the seeds eagerly, as in the supplied prototype.
        for fact in seeds {
            engine.discover(fact, Evidence::Seed);
        }
        for (id, rule) in self.rules.iter().enumerate() {
            if rule.premises.is_empty() {
                engine.stats.rule_firings += 1;
                engine.discover(rule.conclusion, Evidence::Rule(id));
            }
        }
        engine
    }
}

pub struct Derivation<'p, 'r> {
    program: &'p Program<'r>,
    pending: Vec<usize>,
    seen: HashSet<Fact>,
    queue: VecDeque<Fact>,
    stats: Stats,
    evidence: Option<HashMap<Fact, Evidence>>,
}

impl Derivation<'_, '_> {
    fn discover(&mut self, fact: Fact, evidence: Evidence) {
        if self.seen.insert(fact) {
            self.queue.push_back(fact);
            if let Some(proofs) = &mut self.evidence {
                proofs.insert(fact, evidence);
            }
        }
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    pub fn finish(mut self) -> Outcome {
        while self.next().is_some() {}
        Outcome {
            facts: self.seen,
            stats: self.stats,
            evidence: self.evidence,
        }
    }
}

impl Iterator for Derivation<'_, '_> {
    type Item = Fact;

    fn next(&mut self) -> Option<Fact> {
        let fact = self.queue.pop_front()?;
        if let Some(ids) = self.program.dependents.get(&fact) {
            for &id in ids {
                self.stats.dependency_events += 1;
                self.pending[id] -= 1;
                if self.pending[id] == 0 {
                    self.stats.rule_firings += 1;
                    self.discover(self.program.rules[id].conclusion, Evidence::Rule(id));
                }
            }
        }
        Some(fact)
    }
}

impl FusedIterator for Derivation<'_, '_> {}

pub fn worklist(seeds: &[Fact], rules: &[Rule], proofs: bool) -> Outcome {
    Program::new(rules)
        .derive(seeds.iter().copied(), proofs)
        .finish()
}

pub mod relational;
pub mod vectors;
