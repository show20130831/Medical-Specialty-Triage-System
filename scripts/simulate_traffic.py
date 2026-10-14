"""Replay descriptions against a running service, then save its /metrics snapshot.

Used to demonstrate the drift check locally:

    python scripts/simulate_traffic.py --scenario in-distribution --out snapshot.json
    python scripts/simulate_traffic.py --scenario excluded-specialties --out snapshot.json

`in-distribution` replays the prepared test split; `excluded-specialties` replays
descriptions from MTSamples specialties the model was not trained on. Both read local
data only and never print or save description text; the snapshot holds counters only.
Restart the service between scenarios so each snapshot covers one scenario.
"""

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def descriptions(scenario: str, limit: int) -> list[str]:
    if scenario == "in-distribution":
        test = pd.read_csv(REPO_ROOT / "outputs/data/dedup_seed42/test.csv", keep_default_na=False)
        return test["description"].tolist()[:limit]
    taxonomy = yaml.safe_load((REPO_ROOT / "configs/taxonomy.yaml").read_text(encoding="utf-8"))
    known = set(taxonomy["direct_mapping"]) | set(taxonomy["keyword_reclassification_specialties"])
    raw = pd.read_csv(REPO_ROOT / "data/mtsamples.csv", encoding="cp1252",
                      usecols=["description", "medical_specialty"]).dropna()
    excluded = raw[~raw["medical_specialty"].str.strip().isin(known)]
    texts = excluded["description"].str.strip().drop_duplicates()
    return texts[texts.str.len().between(1, 1000)].tolist()[:limit]


def post(base_url: str, text: str) -> None:
    request = urllib.request.Request(
        f"{base_url}/predict", data=json.dumps({"description": text}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request) as response:
        response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=["in-distribution", "excluded-specialties"],
                        required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    texts = descriptions(args.scenario, args.limit)
    for text in texts:
        post(args.base_url, text)
    with urllib.request.urlopen(f"{args.base_url}/metrics") as response:
        snapshot = json.load(response)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"Sent {len(texts)} {args.scenario} requests; snapshot saved to {args.out}")


if __name__ == "__main__":
    main()
