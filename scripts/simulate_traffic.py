"""Replay descriptions against a running service for drift drills.

    python scripts/simulate_traffic.py --scenario in-distribution --total 400
    python scripts/simulate_traffic.py --scenario excluded-specialties

`in-distribution` replays the prepared test split (cycling through it when --total is
larger, as drill D0 needs one 400-prediction window from 398 rows); `excluded-specialties`
replays descriptions from MTSamples specialties the model was not trained on (drill D2).
Both read local data only and never print or save description text. Afterwards the
script prints the service's drift metrics, which hold only the last completed window's
PSI values, uncertain rate, and status. Restart the service before a drill so its
windows hold only drill traffic.
"""

import argparse
import itertools
import json
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def descriptions(scenario: str) -> list[str]:
    if scenario == "in-distribution":
        test = pd.read_csv(REPO_ROOT / "outputs/data/dedup_seed42/test.csv", keep_default_na=False)
        return test["description"].tolist()
    taxonomy = yaml.safe_load((REPO_ROOT / "configs/taxonomy.yaml").read_text(encoding="utf-8"))
    known = set(taxonomy["direct_mapping"]) | set(taxonomy["keyword_reclassification_specialties"])
    raw = pd.read_csv(REPO_ROOT / "data/mtsamples.csv", encoding="cp1252",
                      usecols=["description", "medical_specialty"]).dropna()
    excluded = raw[~raw["medical_specialty"].str.strip().isin(known)]
    texts = excluded["description"].str.strip().drop_duplicates()
    return texts[texts.str.len().between(1, 1000)].tolist()


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
    parser.add_argument("--total", type=int, help="requests to send (default: each text once)")
    args = parser.parse_args()

    texts = descriptions(args.scenario)
    total = args.total or len(texts)
    for text in itertools.islice(itertools.cycle(texts), total):
        post(args.base_url, text)
    with urllib.request.urlopen(f"{args.base_url}/metrics") as response:
        drift = [line for line in response.read().decode().splitlines()
                 if line.startswith("triage_drift") and not line.startswith("#")]
    print(f"Sent {total} {args.scenario} requests. Drift metrics:")
    print("\n".join(drift))


if __name__ == "__main__":
    main()
