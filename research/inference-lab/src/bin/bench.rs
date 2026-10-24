use inference_lab::{Evidence, Outcome, Program, Rule, delta_rounds, naive, relational, worklist};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashSet;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

// Profiling is a separate pass. Timed runs only pay the enabled-flag check.
struct Allocator;
static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn allocated(bytes: usize) {
    ALLOCS.fetch_add(1, Relaxed);
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(live, Relaxed);
}

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if ENABLED.load(Relaxed) && !p.is_null() {
            allocated(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if ENABLED.load(Relaxed) && !p.is_null() {
            allocated(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        if ENABLED.load(Relaxed) {
            LIVE.fetch_sub(layout.size(), Relaxed);
        }
        unsafe {
            System.dealloc(p, layout);
        }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, layout, size) };
        if ENABLED.load(Relaxed) && !q.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            allocated(size);
        }
        q
    }
}

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

fn profile<T>(f: impl FnOnce() -> T) -> (usize, usize) {
    ALLOCS.store(0, Relaxed);
    LIVE.store(0, Relaxed);
    PEAK.store(0, Relaxed);
    ENABLED.store(true, Relaxed);
    let value = f();
    let result = (ALLOCS.load(Relaxed), PEAK.load(Relaxed));
    drop(value);
    ENABLED.store(false, Relaxed);
    assert_eq!(
        LIVE.load(Relaxed),
        0,
        "profile scope must own all allocations"
    );
    result
}

fn stats(o: &Outcome) -> String {
    let s = o.stats;
    format!(
        "\"facts\":{},\"rounds\":{},\"rule_checks\":{},\"premise_checks\":{},\"dependency_events\":{},\"rule_firings\":{}",
        o.facts.len(),
        s.rounds,
        s.rule_checks,
        s.premise_checks,
        s.dependency_events,
        s.rule_firings
    )
}

fn fixtures() -> Vec<(&'static str, Vec<u32>, Vec<Rule>)> {
    let chain = (0..2048).map(|i| Rule::new(vec![i], i + 1)).collect();
    let fanout = (1..=10_000).map(|i| Rule::new(vec![0], i)).collect();
    let mut conjunction = Vec::new();
    for i in 0..3000 {
        conjunction.push(Rule::new(vec![i, i + 1, i + 1], i + 2));
    }
    conjunction.push(Rule::new(vec![3001], 0));
    let mut sparse: Vec<_> = (100..20_100)
        .map(|i| Rule::new(vec![i, i + 30_000], i + 1))
        .collect();
    sparse.extend((0..30).map(|i| Rule::new(vec![i], i + 1)));
    vec![
        ("chain", vec![0], chain),
        ("fanout", vec![0], fanout),
        ("conjunction_cycle", vec![0, 1], conjunction),
        ("mostly_blocked", vec![0], sparse),
    ]
}

fn benchmark(repeats: usize) {
    for (fixture, seeds, rules) in fixtures() {
        let expected = naive(&seeds, &rules).facts;
        let program = Program::new(&rules);
        for variant in [
            "naive",
            "delta_rounds",
            "worklist",
            "worklist_reused_index",
            "worklist_proofs",
        ] {
            let run = || match variant {
                "naive" => naive(&seeds, &rules),
                "delta_rounds" => delta_rounds(&seeds, &rules),
                "worklist_reused_index" => program.derive(seeds.iter().copied(), false).finish(),
                "worklist_proofs" => worklist(&seeds, &rules, true),
                _ => worklist(&seeds, &rules, false),
            };
            assert_eq!(run().facts, expected);
            let (allocations, peak_heap_bytes) = profile(run);
            for repetition in 0..repeats {
                let start = Instant::now();
                let output = black_box(run());
                let elapsed_ns = start.elapsed().as_nanos();
                assert_eq!(output.facts, expected);
                println!(
                    "{{\"suite\":\"ground_closure\",\"fixture\":\"{fixture}\",\"variant\":\"{variant}\",\"repetition\":{repetition},\"elapsed_ns\":{elapsed_ns},\"rules\":{},\"allocations\":{allocations},\"peak_heap_bytes\":{peak_heap_bytes},{}}}",
                    rules.len(),
                    stats(&output)
                );
            }
        }
    }
    for (fixture, edges) in [
        ("path", (0..160).map(|i| (i, i + 1)).collect::<Vec<_>>()),
        ("cycle", (0..96).map(|i| (i, (i + 1) % 96)).collect()),
    ] {
        let expected = relational::reachability(&edges, false).paths;
        for semi_naive in [false, true] {
            let variant = if semi_naive {
                "semi_naive_indexed_join"
            } else {
                "naive_indexed_join"
            };
            let (allocations, peak_heap_bytes) =
                profile(|| relational::reachability(&edges, semi_naive));
            for repetition in 0..repeats {
                let start = Instant::now();
                let o = black_box(relational::reachability(&edges, semi_naive));
                let elapsed_ns = start.elapsed().as_nanos();
                assert_eq!(o.paths, expected);
                println!(
                    "{{\"suite\":\"relational_closure\",\"fixture\":\"{fixture}\",\"variant\":\"{variant}\",\"repetition\":{repetition},\"elapsed_ns\":{elapsed_ns},\"facts\":{},\"rounds\":{},\"join_probes\":{},\"join_matches\":{},\"allocations\":{allocations},\"peak_heap_bytes\":{peak_heap_bytes}}}",
                    o.paths.len(),
                    o.rounds,
                    o.join_probes,
                    o.join_matches
                );
            }
        }
    }
}

fn infer(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let data = std::fs::read_to_string(path)?;
    let mut seeds = Vec::new();
    let mut rules = Vec::new();
    for line in data.lines().filter(|x| !x.trim().is_empty()) {
        let mut parts = line.split_whitespace();
        let tag = parts.next().ok_or("missing tag")?;
        let ids: Vec<u32> = parts.map(str::parse).collect::<Result<_, _>>()?;
        match (tag, ids.as_slice()) {
            ("s", [id]) => seeds.push(*id),
            ("r", [head, premises @ ..]) => rules.push(Rule::new(premises.to_vec(), *head)),
            _ => return Err(format!("invalid program line: {line}").into()),
        }
    }
    let start = Instant::now();
    let o = worklist(&seeds, &rules, true);
    let elapsed_ns = start.elapsed().as_nanos();
    let mut facts: Vec<_> = o.facts.iter().copied().collect();
    facts.sort_unstable();
    let seed_set: HashSet<_> = seeds.into_iter().collect();
    let evidence = o.evidence.as_ref().unwrap();
    for fact in &facts {
        match evidence[fact] {
            Evidence::Seed => assert!(seed_set.contains(fact)),
            Evidence::Rule(id) => {
                assert_eq!(rules[id].conclusion, *fact);
                assert!(rules[id].premises.iter().all(|p| o.facts.contains(p)));
            }
        }
    }
    let proof_pairs: Vec<_> = facts
        .iter()
        .map(|f| match evidence[f] {
            Evidence::Seed => format!("[{f},null]"),
            Evidence::Rule(id) => format!("[{f},{id}]"),
        })
        .collect();
    println!(
        "{{\"fact_ids\":{facts:?},\"proofs\":[{}],\"elapsed_ns\":{elapsed_ns},{}}}",
        proof_pairs.join(","),
        stats(&o)
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [mode, path] if mode == "infer" => infer(path)?,
        [mode, count] if mode == "bench" => benchmark(count.parse()?),
        [] => benchmark(5),
        _ => return Err("usage: bench [bench REPEATS | infer PROGRAM.txt]".into()),
    }
    Ok(())
}
