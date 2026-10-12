"""Map MTSamples specialties to the eight served specialty groups."""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Taxonomy:
    direct_mapping: dict[str, str]
    keyword_specialties: frozenset[str]
    keyword_weights: dict[str, dict[str, int]]

    @property
    def labels(self) -> tuple[str, ...]:
        """Served label order: sorted group names, matching the model's id2label."""
        return tuple(sorted(set(self.direct_mapping.values())))


def load_taxonomy(path: Path) -> Taxonomy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    taxonomy = Taxonomy(
        direct_mapping=raw["direct_mapping"],
        keyword_specialties=frozenset(raw["keyword_reclassification_specialties"]),
        keyword_weights=raw["keyword_weights"],
    )
    if set(taxonomy.keyword_weights) != set(taxonomy.labels):
        raise ValueError("keyword_weights must define exactly the direct-mapping groups.")
    return taxonomy


def clean_keywords(text: object) -> str:
    """Drop the trailing "NOTE: ..." disclaimer MTSamples appends to some keyword fields."""
    if not isinstance(text, str) or not text.strip():
        return ""
    return re.sub(r"NOTE.*", "", text, flags=re.IGNORECASE).lower().strip()


def label_from_keywords(keywords: str, taxonomy: Taxonomy) -> str | None:
    """Return the highest weighted group, or None when no keyword matches."""
    if not keywords:
        return None
    scores = {
        label: sum(weight for term, weight in weights.items() if term in keywords)
        for label, weights in taxonomy.keyword_weights.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def assign_label(specialty: str, keywords: str, taxonomy: Taxonomy) -> tuple[str | None, str]:
    """Return (label, source) where source is "direct", "keyword", or "unmapped"."""
    if specialty in taxonomy.direct_mapping:
        return taxonomy.direct_mapping[specialty], "direct"
    if specialty in taxonomy.keyword_specialties:
        return label_from_keywords(clean_keywords(keywords), taxonomy), "keyword"
    return None, "unmapped"
