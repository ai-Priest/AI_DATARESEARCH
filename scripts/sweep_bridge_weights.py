"""
Bridge Weight Sweep — NDCG@3 optimization
Sweeps NEURAL_WEIGHT / SEMANTIC_WEIGHT / KEYWORD_WEIGHT combinations in
src/ai/neural_ai_bridge.py against the validation split NDCG@3 score.

Usage:
    python scripts/sweep_bridge_weights.py

Outputs best weights to stdout and optionally patches neural_ai_bridge.py.
"""

import sys
import os
import json
import itertools
import numpy as np
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def ndcg_at_k(scores: list, relevances: list, k: int = 3) -> float:
    """Compute NDCG@k for a single query."""
    if len(scores) < k:
        return 0.0
    order = np.argsort(scores)[::-1]
    top_k_rel = np.array(relevances)[order[:k]]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(top_k_rel))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def load_validation_samples(data_path: Path) -> list:
    """Load validation samples from the boosted training data."""
    with open(data_path) as f:
        data = json.load(f)

    # Handle pre-split format {"train": [...], "validation": [...], "test": [...]}
    if isinstance(data, dict) and "validation" in data:
        return data["validation"]
    samples = data if isinstance(data, list) else data.get("samples", [])
    # Use last 15% as validation proxy (mirrors dl_pipeline split)
    split_idx = int(len(samples) * 0.85)
    return samples[split_idx:]


def score_neural(sample: dict) -> float:
    """Proxy neural score: use existing neural_score field or fall back to label."""
    return float(sample.get("neural_score", sample.get("relevance_score", sample.get("label", 0.0))))


def score_semantic(sample: dict) -> float:
    """Proxy semantic score: use semantic_score field or fall back to relevance."""
    return float(sample.get("semantic_score", sample.get("relevance_score", sample.get("label", 0.0))))


def score_keyword(sample: dict) -> float:
    """Proxy keyword score: use keyword_score field or compute simple title match."""
    if "keyword_score" in sample and sample["keyword_score"] is not None:
        return float(sample["keyword_score"])
    query = str(sample.get("query", "")).lower()
    # dataset_title is the field name in this project's data format
    title = str(sample.get("dataset_title", sample.get("title", ""))).lower()
    words = query.split()
    if not words:
        return 0.0
    return sum(1 for w in words if w in title) / len(words)


def evaluate_weights(val_samples: list, nw: float, sw: float, kw: float) -> float:
    """Compute mean NDCG@3 across all queries for given weight combination."""
    # Group samples by query
    query_groups: dict[str, list] = {}
    for s in val_samples:
        q = str(s.get("query", s.get("original_query", "unknown")))
        query_groups.setdefault(q, []).append(s)

    ndcg_scores = []
    for query, samples in query_groups.items():
        if len(samples) < 3:
            continue
        combined = [
            nw * score_neural(s) + sw * score_semantic(s) + kw * score_keyword(s)
            for s in samples
        ]
        relevances = [float(s.get("relevance_score", s.get("label", 0.0))) for s in samples]
        ndcg_scores.append(ndcg_at_k(combined, relevances, k=3))

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def patch_bridge_weights(neural: float, semantic: float, keyword: float):
    """Update the hardcoded weights in neural_ai_bridge.py."""
    bridge_path = PROJECT_ROOT / "src" / "ai" / "neural_ai_bridge.py"
    text = bridge_path.read_text()

    import re
    text = re.sub(r"NEURAL_WEIGHT\s*=\s*[\d.]+", f"NEURAL_WEIGHT = {neural:.3f}", text)
    text = re.sub(r"SEMANTIC_WEIGHT\s*=\s*[\d.]+", f"SEMANTIC_WEIGHT = {semantic:.3f}", text)
    text = re.sub(r"KEYWORD_WEIGHT\s*=\s*[\d.]+", f"KEYWORD_WEIGHT = {keyword:.3f}", text)

    bridge_path.write_text(text)
    print(f"\n✓ Patched neural_ai_bridge.py with best weights.")


def main():
    data_path = PROJECT_ROOT / "data" / "processed" / "enhanced_training_data_boosted.json"
    if not data_path.exists():
        # Fall back to graded data
        data_path = PROJECT_ROOT / "data" / "processed" / "enhanced_training_data_graded.json"

    print(f"Loading validation samples from: {data_path.name}")
    val_samples = load_validation_samples(data_path)
    print(f"Validation samples: {len(val_samples)}")

    # Current baseline
    baseline = evaluate_weights(val_samples, 0.6, 0.25, 0.15)
    print(f"\nBaseline (0.6 / 0.25 / 0.15): NDCG@3 = {baseline:.4f}")

    # Grid: weights must sum to 1.0, each in [0.05, 0.80] step 0.05
    steps = [round(x * 0.05, 2) for x in range(1, 17)]  # 0.05 to 0.80
    candidates = [
        (nw, sw, kw)
        for nw, sw, kw in itertools.product(steps, repeat=3)
        if abs(nw + sw + kw - 1.0) < 0.001
    ]
    print(f"Testing {len(candidates)} weight combinations...\n")

    results = []
    for i, (nw, sw, kw) in enumerate(candidates):
        score = evaluate_weights(val_samples, nw, sw, kw)
        results.append((score, nw, sw, kw))
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(candidates)} — best so far: {max(r[0] for r in results):.4f}")

    results.sort(reverse=True)

    print("\n=== Top 10 weight combinations ===")
    print(f"{'NDCG@3':>8}  {'Neural':>7}  {'Semantic':>9}  {'Keyword':>8}")
    print("-" * 40)
    for score, nw, sw, kw in results[:10]:
        marker = " ← BEST" if (nw, sw, kw) == (results[0][1], results[0][2], results[0][3]) else ""
        print(f"{score:>8.4f}  {nw:>7.3f}  {sw:>9.3f}  {kw:>8.3f}{marker}")

    best_score, best_nw, best_sw, best_kw = results[0]
    improvement = (best_score - baseline) / max(baseline, 1e-9) * 100

    print(f"\n✓ Best: NEURAL={best_nw}, SEMANTIC={best_sw}, KEYWORD={best_kw}")
    print(f"  NDCG@3: {baseline:.4f} → {best_score:.4f} (+{improvement:.1f}%)")

    # Save results
    output = {
        "baseline": {"weights": [0.6, 0.25, 0.15], "ndcg_at_3": baseline},
        "best": {"weights": [best_nw, best_sw, best_kw], "ndcg_at_3": best_score},
        "improvement_pct": improvement,
        "top_10": [
            {"ndcg_at_3": s, "neural": n, "semantic": se, "keyword": k}
            for s, n, se, k in results[:10]
        ]
    }
    out_path = PROJECT_ROOT / "outputs" / "bridge_weight_sweep_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved → {out_path}")

    # Ask before patching
    if improvement > 0.5:
        answer = input(f"\nPatch neural_ai_bridge.py with best weights? [y/N]: ").strip().lower()
        if answer == "y":
            patch_bridge_weights(best_nw, best_sw, best_kw)
    else:
        print("\nNo significant improvement found — current weights are near-optimal.")


if __name__ == "__main__":
    main()
