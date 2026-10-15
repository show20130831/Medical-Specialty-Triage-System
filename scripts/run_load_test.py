"""Run the k6 load steps against the Compose stack and write a capacity report.

    python scripts/run_load_test.py --steps 1 2 4 8 --duration 3m --out reports

Each step runs k6 (Compose profile `loadtest`) with a fixed number of virtual users
against the API container, samples the container's CPU with `docker stats`, and reads
k6's summary export. The highest step meeting the latency and error targets is the
supported capacity (docs/operations-plan.md §6 and §9). Payloads are synthetic.
"""

import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
P95_TARGET_MS = 500
ERROR_TARGET = 0.01


def api_container() -> str:
    return subprocess.run(["docker", "compose", "ps", "-q", "api"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def sample_cpu(container: str, stop: threading.Event, samples: list[float]) -> None:
    while not stop.is_set():
        out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}",
                              container], capture_output=True, text=True).stdout.strip()
        if out.endswith("%"):
            samples.append(float(out[:-1]))
        stop.wait(5)


def run_step(vus: int, duration: str, label: str, sample: bool = True) -> dict:
    export = f"/results/{label}.json"
    container = api_container()
    stop, samples = threading.Event(), []
    # `docker stats` itself costs CPU on Docker Desktop; --no-cpu measures without it.
    sampler = threading.Thread(target=sample_cpu, args=(container, stop, samples), daemon=True)
    if sample:
        sampler.start()
    started = time.time()
    subprocess.run(["docker", "compose", "--profile", "loadtest", "run", "--rm", "k6", "run",
                    "--quiet", "-e", f"VUS={vus}", "-e", f"DURATION={duration}",
                    f"--summary-export={export}", "/loadtest/load.js"],
                   cwd=REPO_ROOT, check=True)
    stop.set()
    if sample:
        sampler.join()
    summary = json.loads((REPO_ROOT / "outputs/loadtest" / f"{label}.json").read_text())
    metrics = summary["metrics"]
    duration_ms = metrics["http_req_duration"]
    failed = metrics.get("http_req_failed", {}).get("value", 0.0)
    return {
        "label": label, "vus": vus, "seconds": round(time.time() - started),
        "requests": metrics["http_reqs"]["count"],
        "throughput_rps": round(metrics["http_reqs"]["rate"], 2),
        "p50_ms": round(duration_ms["med"], 1), "p95_ms": round(duration_ms["p(95)"], 1),
        "p99_ms": round(duration_ms["p(99)"], 1), "max_ms": round(duration_ms["max"], 1),
        "error_rate": failed,
        "cpu_percent_median": round(statistics.median(samples), 1) if samples else None,
        "meets_targets": duration_ms["p(95)"] <= P95_TARGET_MS and failed < ERROR_TARGET,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--duration", default="3m")
    parser.add_argument("--label-prefix", default="step")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports")
    parser.add_argument("--no-cpu", action="store_true", help="skip docker stats sampling")
    args = parser.parse_args()

    (REPO_ROOT / "outputs/loadtest").mkdir(parents=True, exist_ok=True)
    results = []
    for vus in args.steps:
        result = run_step(vus, args.duration, f"{args.label_prefix}_{vus}vu", not args.no_cpu)
        print(json.dumps(result), flush=True)
        results.append(result)
    passing = [r["vus"] for r in results if r["meets_targets"]]
    out = args.out / f"load_test_{args.label_prefix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"p95_target_ms": P95_TARGET_MS, "error_target": ERROR_TARGET,
                               "steps": results,
                               "highest_passing_vus": max(passing) if passing else None},
                              indent=2) + "\n", encoding="utf-8")
    print("highest step meeting targets:", max(passing) if passing else None)


if __name__ == "__main__":
    main()
