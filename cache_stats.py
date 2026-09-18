#!/usr/bin/env python3
"""Aggregate SGLang scheduler prefill/decode stats from a head log.

Usage: python3 cache_stats.py [log_path] [lines]
Prints per-request cache hit rate and generated-token totals per hour.
"""
import re
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "sglang_head.log"
tail = int(sys.argv[2]) if len(sys.argv) > 2 else 0

prefill_re = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}):\d{2} TP0\] Prefill batch, "
    r"#new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), "
    r"token usage: ([\d.]+)"
)
decode_re = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) TP0\] Decode batch")
gen_re = re.compile(r"gen throughput \(token/s\): ([\d.]+)")

with open(path, errors="replace") as f:
    lines = f.readlines()
if tail:
    lines = lines[-tail:]

tot_new = tot_cached = tot_seqs = tot_batches = 0
by_hour = defaultdict(lambda: {"new": 0, "cached": 0, "seqs": 0})
for line in lines:
    m = prefill_re.search(line)
    if not m:
        continue
    ts, seqs, new, cached = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    tot_new += new
    tot_cached += cached
    tot_seqs += seqs
    tot_batches += 1
    h = by_hour[ts[:13]]
    h["new"] += new
    h["cached"] += cached
    h["seqs"] += seqs

total = tot_new + tot_cached
print(f"log: {path} ({len(lines)} lines, {tot_batches} prefill batches)")
if not total:
    print("no prefill batches found")
    sys.exit(0)
print(f"seqs prefilled:      {tot_seqs:>10,}")
print(f"new tokens:          {tot_new:>10,}")
print(f"cached tokens:       {tot_cached:>10,}")
print(f"total prefill tokens:{total:>10,}")
print(f"CACHE HIT RATE:      {tot_cached / total * 100:.1f}%")
print()
print(f"{'hour':<17}{'new':>10}{'cached':>10}{'hit rate':>10}{'seqs':>8}")
for hour in sorted(by_hour):
    h = by_hour[hour]
    t = h["new"] + h["cached"]
    print(f"{hour:<17}{h['new']:>10,}{h['cached']:>10,}{h['cached'] / t * 100:>9.1f}%{h['seqs']:>8}")
