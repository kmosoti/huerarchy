use inference_lab::vectors::{Index, Kernel, Matrix};
use std::time::Instant;

fn memory() -> (usize, usize) {
    let data = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let value = |key: &str| {
        data.lines()
            .find(|line| line.starts_with(key))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            * 1024
    };
    (value("VmRSS:"), value("VmHWM:"))
}

fn scheduled_cpu_ns() -> Option<u64> {
    // Linux reports scheduled runtime for this thread. The worker is single-threaded.
    std::fs::read_to_string("/proc/self/schedstat")
        .ok()?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: vector-search CORPUS.vec QUERIES.vec TOP_K".into());
    }
    let start = Instant::now();
    let index = Index::new(Matrix::read(&args[0])?)?;
    let queries = Matrix::read(&args[1])?;
    let k = args[2].parse::<usize>()?;
    if queries.columns != index.matrix().columns {
        return Err("query/corpus dimensions differ".into());
    }
    let load_ns = start.elapsed().as_nanos();
    let (loaded_rss, _) = memory();
    let kernel = Kernel::detect();
    // Resolve/load everything before timing. One query warms the search code.
    index.search(queries.float_row(0)?, k, kernel)?;
    let cpu_start = scheduled_cpu_ns();
    let loop_start = Instant::now();
    let mut total_ns = 0;
    for row in 0..queries.rows {
        let query = queries.float_row(row)?;
        let tick = Instant::now();
        let matches = index.search(query, k, kernel)?;
        let elapsed_ns = tick.elapsed().as_nanos();
        total_ns += elapsed_ns;
        let ids: Vec<_> = matches.iter().map(|m| m.id.to_string()).collect();
        let scores: Vec<_> = matches.iter().map(|m| m.score.to_string()).collect();
        println!(
            "{{\"query\":{row},\"elapsed_ns\":{elapsed_ns},\"ids\":[{}],\"scores\":[{}]}}",
            ids.join(","),
            scores.join(",")
        );
    }
    let loop_wall_ns = loop_start.elapsed().as_nanos();
    let loop_cpu_ns = scheduled_cpu_ns()
        .zip(cpu_start)
        .and_then(|(end, start)| end.checked_sub(start))
        .filter(|&elapsed| elapsed > 0)
        .map_or_else(|| "null".to_owned(), |n| n.to_string());
    let (rss, peak) = memory();
    println!(
        "{{\"summary\":true,\"kernel\":\"{}\",\"threads\":1,\"rows\":{},\"dimensions\":{},\"queries\":{},\"load_ns\":{load_ns},\"total_search_ns\":{total_ns},\"query_loop_wall_ns\":{loop_wall_ns},\"query_loop_cpu_ns\":{loop_cpu_ns},\"corpus_bytes\":{},\"query_bytes\":{},\"loaded_rss_bytes\":{loaded_rss},\"final_rss_bytes\":{rss},\"peak_rss_bytes\":{peak}}}",
        kernel.name(),
        index.matrix().rows,
        index.matrix().columns,
        queries.rows,
        index.corpus_bytes(),
        queries.payload_bytes()
    );
    Ok(())
}
