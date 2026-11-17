"""Select generators from a packaged, versioned comparison table."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

AUTO_MODEL = "auto"
HAIKU_MODEL = "anthropic/claude-haiku-5.5"
QWEN_MODEL = "qwen/qwen3.8-27b"
QWEN_MARGIN = 0.5
POLICY = "table_haiku_preferred_0.5_v1"
SELECTION_FIELDS = (
    "generator_selection_policy", "generator_selection_basis",
    "generator_selection_qwen_advantage", "generator_selection_table_sha256",
)


@lru_cache(maxsize=1)
def _table():
    raw = files("clir_bench.core").joinpath("generation_model_table.json").read_bytes()
    table = json.loads(raw)
    entries = {(row["corpus"], row["mode"], row["language"]): row for row in table["rows"]}
    return entries, hashlib.sha256(raw).hexdigest()


def selection_policy():
    """Pin the table and decision rule in resumable run fingerprints."""
    return {"policy": POLICY, "table_sha256": _table()[1], "qwen_margin": QWEN_MARGIN}


@dataclass(frozen=True)
class GeneratorSelection:
    model: str
    basis: str
    qwen_advantage: float | None = None

    def metadata(self):
        if self.basis == "explicit":
            return {}
        return dict(zip(SELECTION_FIELDS, (
            POLICY, self.basis,
            self.qwen_advantage if self.qwen_advantage is not None else "",
            _table()[1],
        )))


def select_generator(corpus: str, mode: str, language: str, *, model: str = AUTO_MODEL):
    """Use Qwen only at an advantage of at least 0.5; default missing cells to Haiku.

    Match the comparison analysis: use shared-input differences with at least
    five independent parent documents, otherwise the observed target means.
    A table-approved alternative to both preferred models retains its route.
    """
    if model != AUTO_MODEL:
        return GeneratorSelection(model, "explicit")
    if mode == "practitioner":
        mode = "practitioners"
    row = _table()[0].get((corpus, mode, language))
    if row is None:
        return GeneratorSelection(HAIKU_MODEL, "no_data")
    if row["recommended_model"] not in (HAIKU_MODEL, QWEN_MODEL):
        return GeneratorSelection(row["recommended_model"], "table_alternative")
    if row["paired_documents"] >= 5 and row["haiku_minus_qwen_paired"] is not None:
        advantage = -row["haiku_minus_qwen_paired"]
        basis = "paired_target_mean"
    elif row["haiku_mean"] is not None and row["qwen_mean"] is not None:
        advantage = row["qwen_mean"] - row["haiku_mean"]
        basis = "observed_target_mean"
    else:
        return GeneratorSelection(HAIKU_MODEL, "missing_preferred_scores")
    # Tolerate only arithmetic rounding at the exact half-point boundary.
    selected = QWEN_MODEL if advantage >= QWEN_MARGIN - 1e-9 else HAIKU_MODEL
    return GeneratorSelection(selected, basis, advantage)


def replay_models(models):
    """Expand the selector to provider model IDs when importing response caches."""
    return list(dict.fromkeys(actual for model in models
                             for actual in ((HAIKU_MODEL, QWEN_MODEL) if model == AUTO_MODEL
                                            else (model,))))
