from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections import deque
from pathlib import Path

import numpy as np

from .common import DATA, canonical, digest, emit, file_hash, write_json

HEALTH = ("healthy", "unhealthy", "unknown")
ROLES = ("database", "storage", "web", "queue", "cache")
ISSUES = ("disk full", "memory pressure", "packet loss", "high latency", "processor saturation", "certificate expired")
ROLE_PARAPHRASES = ("relational data", "persistent file", "HTTP application", "message delivery", "temporary lookup")
ISSUE_PARAPHRASES = ("exhausted disk capacity", "insufficient RAM", "dropped network traffic", "slow responses", "CPU overload", "outdated TLS credentials")


def node_attributes(node):
    # Stable attributes; not fitted to evaluation questions.
    return {"node": node, "group": node // 5, "zone": node % 10,
            "rack": (node // 10) % 25, "role": ROLES[(node // 250) % 5]}


def adjacency(nodes):
    # Bounded cyclic components prevent an accidental all-pairs memory explosion.
    result = {i: [] for i in range(nodes)}
    for node in range(nodes):
        start = node // 25 * 25
        size = min(25, nodes - start)
        result[node] = sorted({start + (node - start + step) % size for step in (1, 5)})
    return result


def reachable_reference(graph, source, depth=4):
    frontier = deque([(source, 0)])
    visited = {source}
    while frontier:
        node, level = frontier.popleft()
        if level == depth:
            continue
        for child in graph[node]:
            if child not in visited:
                visited.add(child)
                frontier.append((child, level + 1))
    visited.remove(source)
    return sorted(visited)


def impacted_reference(graph, unhealthy):
    impacted = set(map(int, unhealthy))
    frontier = deque(impacted)
    while frontier:
        node = frontier.popleft()
        for child in graph[node]:
            if child not in impacted:
                impacted.add(child)
                frontier.append(child)
    return impacted


def decision_reference(states, node):
    group = list(map(int, states[node // 5 * 5:node // 5 * 5 + 5]))
    if len(group) != 5 or 2 in group:
        return "unknown"
    remaining = sum(code == 0 for code in group) - (int(states[node]) == 0)
    return "permit" if remaining >= 3 else "block"


def summary(row):
    return (f"Node n{row['node']:05d} is a {row['role']} server in zone z{row['zone']:02d}, "
            f"rack r{row['rack']:02d}. Health: {row['health']}. Reported issue: {row['issue']}. "
            f"Replica group g{row['group']:04d}.")


def generate(directory=DATA, *, events=100_000, nodes=5_000, queries=10_000, seed=42):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if nodes % 5 or nodes < 25 or events < nodes or queries < 4 or queries % 4:
        raise ValueError("nodes must be >=25 and divisible by 5; events>=nodes; queries divisible by 4")
    rng = random.Random(seed)
    per_epoch = max(1, events // 100)
    epochs = (events + per_epoch - 1) // per_epoch
    states = np.full(nodes, 2, dtype=np.int8)
    snapshots = []
    latest_event = np.full(nodes, -1, dtype=np.int64)
    snapshot_sources = []
    latest_issue = np.zeros(nodes, dtype=np.int8)
    graph = adjacency(nodes)
    path = directory / "events.sqlite"
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    db.executescript("""
      PRAGMA journal_mode=DELETE;
      CREATE TABLE events(id INTEGER PRIMARY KEY,epoch INTEGER,node INTEGER,health INTEGER,issue INTEGER,payload BLOB NOT NULL,sha256 TEXT NOT NULL);
      CREATE TABLE nodes(node INTEGER PRIMARY KEY,group_id INTEGER,health INTEGER,issue INTEGER,zone INTEGER,rack INTEGER,role TEXT,summary TEXT);
      CREATE TABLE edges(source INTEGER,target INTEGER,PRIMARY KEY(source,target)) WITHOUT ROWID;
      CREATE TABLE snapshots(epoch INTEGER,node INTEGER,health INTEGER,PRIMARY KEY(epoch,node)) WITHOUT ROWID;
      CREATE VIRTUAL TABLE summaries USING fts5(text, tokenize='unicode61');
    """)
    pending = []
    source_hashes = []
    with (directory / "events.jsonl").open("w") as stream:
        for event_id in range(events):
            node = event_id if event_id < nodes else rng.randrange(nodes)
            health = 0 if event_id < nodes else rng.choices((0, 1, 2), weights=(65, 30, 5))[0]
            issue = rng.randrange(len(ISSUES))
            epoch = event_id // per_epoch
            event = {"id": event_id, "epoch": epoch, "node": node,
                     "health": HEALTH[health], "issue": ISSUES[issue],
                     "source": f"monitor-{node % 4}", "timestamp_ms": 1_800_000_000_000 + event_id * 100}
            raw = canonical(event)
            sha = hashlib.sha256(raw).hexdigest()
            source_hashes.append(sha)
            stream.write(raw.decode() + "\n")
            pending.append((event_id, epoch, node, health, issue, raw, sha))
            states[node] = health
            latest_event[node] = event_id
            latest_issue[node] = issue
            if (event_id + 1) % per_epoch == 0 or event_id + 1 == events:
                db.executemany("INSERT INTO events VALUES(?,?,?,?,?,?,?)", pending)
                pending.clear()
                snapshots.append(states.copy())
                snapshot_sources.append(latest_event.copy())
                db.executemany("INSERT INTO snapshots VALUES(?,?,?)", ((epoch, i, int(h)) for i, h in enumerate(states)))
        db.executemany("INSERT INTO edges VALUES(?,?)", ((a, b) for a in graph for b in graph[a]))
        docs = []
        for node in range(nodes):
            row = dict(node_attributes(node), health=HEALTH[int(states[node])], issue=ISSUES[int(latest_issue[node])])
            row["text"] = summary(row)
            docs.append(row)
            db.execute("INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?)", (node, row["group"], int(states[node]), int(latest_issue[node]), row["zone"], row["rack"], row["role"], row["text"]))
            db.execute("INSERT INTO summaries(rowid,text) VALUES(?,?)", (node + 1, row["text"]))
        db.executescript("CREATE INDEX events_node ON events(node,id); CREATE INDEX nodes_group ON nodes(group_id); ANALYZE;")
        db.commit()
    db.close()
    snapshots = np.array(snapshots, dtype=np.int8)
    np.save(directory / "snapshots.npy", snapshots)
    np.save(directory / "snapshot-sources.npy", np.array(snapshot_sources, dtype=np.int64))
    with (directory / "documents.jsonl").open("w") as stream:
        for doc in docs:
            emit(stream, doc)
    relevant = {}
    for doc in docs:
        key = (doc["zone"], doc["rack"], doc["role"], doc["issue"])
        relevant.setdefault(key, []).append(doc["node"])
    templates = (
        "Find {role} hosts in zone z{zone:02d}, rack r{rack:02d}, experiencing {issue}.",
        "Which {role} machines have {issue} in rack r{rack:02d} of zone z{zone:02d}?",
        "Locate servers for {role} with {issue}; zone z{zone:02d}, rack r{rack:02d}.",
        "Show affected {role} systems: {issue}, rack r{rack:02d}, zone z{zone:02d}.",
    )
    with (directory / "queries.jsonl").open("w") as stream:
        for family in ("reconstruct", "relational", "retrieval", "decision"):
            for i in range(queries // 4):
                node = rng.randrange(nodes)
                q = {"family": family, "id": f"{family}-{i}", "node": node}
                if family == "reconstruct":
                    event_id = rng.randrange(events)
                    q.update(event_id=event_id, expected_sha256=source_hashes[event_id])
                elif family == "relational":
                    q.update(depth=4, expected=reachable_reference(graph, node))
                elif family == "decision":
                    epoch = rng.randrange(epochs)
                    q.update(epoch=epoch, expected=decision_reference(snapshots[epoch], node))
                else:
                    doc = docs[node]
                    role = ROLE_PARAPHRASES[ROLES.index(doc["role"])]
                    issue = ISSUE_PARAPHRASES[ISSUES.index(doc["issue"])]
                    q["text"] = templates[i % len(templates)].format(role=role, issue=issue, **{k: doc[k] for k in ("zone", "rack")})
                    q["expected"] = relevant[(doc["zone"], doc["rack"], doc["role"], doc["issue"])]
                emit(stream, q)
    context = {"schema": 1, "rules": "positive-ground-v1", "entities": [node_attributes(n) for n in range(nodes)],
               "edges": [[a, b] for a in graph for b in graph[a]]}
    write_json(directory / "context.json", context)
    manifest = {"seed": seed, "events": events, "nodes": nodes, "queries": queries,
                "epochs": epochs, "events_per_epoch": per_epoch,
                "query_templates_held_out_from_documents": True,
                "source_sha256": file_hash(directory / "events.jsonl"),
                "documents_sha256": file_hash(directory / "documents.jsonl"),
                "queries_sha256": file_hash(directory / "queries.jsonl"),
                "snapshot_sha256": file_hash(directory / "snapshots.npy"),
                "snapshot_sources_sha256": file_hash(directory / "snapshot-sources.npy"),
                "context_hash": digest(context),
                "source_provenance_retained": True}
    write_json(directory / "manifest.json", manifest)
    return manifest


def sql_reachable(db, node, depth=4):
    return [r[0] for r in db.execute("""WITH RECURSIVE reach(node,depth) AS (
      SELECT ?,0 UNION SELECT edges.target,reach.depth+1 FROM edges JOIN reach ON edges.source=reach.node WHERE reach.depth<?
    ) SELECT DISTINCT node FROM reach WHERE node<>? ORDER BY node""", (node, depth, node))]


def sql_decision(db, epoch, node):
    start = node // 5 * 5
    rows = db.execute("SELECT node,health FROM snapshots WHERE epoch=? AND node BETWEEN ? AND ? ORDER BY node", (epoch, start, start + 4)).fetchall()
    if len(rows) != 5 or any(h == 2 for _, h in rows):
        return "unknown"
    healthy = sum(h == 0 for _, h in rows)
    target_health = next(h for n, h in rows if n == node)
    return "permit" if healthy - (target_health == 0) >= 3 else "block"


def lexical_search(db, text, limit=10):
    import re
    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    stop = {"a", "the", "in", "of", "for", "with", "which", "have", "find", "show", "locate", "servers", "systems", "hosts", "machines", "experiencing", "affected"}
    expression = " OR ".join('"' + t + '"' for t in tokens if t.lower() not in stop)
    if not expression:
        return []
    return [r[0] - 1 for r in db.execute("SELECT rowid FROM summaries WHERE summaries MATCH ? ORDER BY bm25(summaries),rowid LIMIT ?", (expression, limit))]
