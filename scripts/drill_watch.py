"""Watch Prometheus alerts during a drill and print state changes with elapsed time.

    python scripts/drill_watch.py --until TriageDriftWarning --timeout 600

Polls the Prometheus alerts API every 5 seconds from the moment it starts (the drill's
start time) and prints each alert's transition to pending, firing, or resolved, with
seconds since start. Exits 0 when every alert named by --until is firing, or 1 on
timeout, so the detection time in a drill record comes from this log.
"""

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone


def active_alerts(prometheus: str) -> dict[str, str]:
    with urllib.request.urlopen(f"{prometheus}/api/v1/alerts", timeout=5) as response:
        alerts = json.load(response)["data"]["alerts"]
    return {a["labels"]["alertname"]: a["state"] for a in alerts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prometheus", default="http://127.0.0.1:9090")
    parser.add_argument("--until", action="append", default=[],
                        help="alert that must be firing to stop (repeatable)")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()

    started = time.monotonic()
    print(f"drill start {datetime.now(timezone.utc).isoformat(timespec='seconds')}", flush=True)
    seen: dict[str, str] = {}
    while time.monotonic() - started < args.timeout:
        current = active_alerts(args.prometheus)
        for name in sorted(set(seen) | set(current)):
            state = current.get(name, "resolved")
            if seen.get(name) != state and not (name not in seen and state == "resolved"):
                elapsed = time.monotonic() - started
                print(f"+{elapsed:6.0f}s  {name:32s} {state}", flush=True)
                seen[name] = state
        if args.until and all(current.get(name) == "firing" for name in args.until):
            sys.exit(0)
        time.sleep(5)
    print(f"timeout after {args.timeout:.0f}s", flush=True)
    sys.exit(1 if args.until else 0)


if __name__ == "__main__":
    main()
