import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from BayesNet import BayesNet

# -----------------------------
# 1) A small default Fraud Bayes Net
# -----------------------------
#
# Variables:
# - AmountHigh:   Is the transaction amount unusually high? (T/F)
# - NewDevice:    Is this a new / unseen device? (T/F)
# - IPMismatch:   Does IP location mismatch billing / expected region? (T/F)
# - PastChargeback: Has this user had a past chargeback? (T/F)
# - Fraud:        Is this likely fraud? (T/F)
#
# Structure:
# Parents(Fraud) = [AmountHigh, NewDevice, IPMismatch, PastChargeback]
#
# Tables format matches your BayesNet.py:
# tables[var] = [
#   [ [parent_values_in_order], [probabilities_in_same_order_as_nodes[var]] ],
#   ...
# ]


def build_default_fraud_net_json() -> Dict[str, Any]:
    nodes = {
        "AmountHigh": ["F", "T"],
        "NewDevice": ["F", "T"],
        "IPMismatch": ["F", "T"],
        "PastChargeback": ["F", "T"],
        "Fraud": ["F", "T"],
    }

    parents = {
        "AmountHigh": [],
        "NewDevice": [],
        "IPMismatch": [],
        "PastChargeback": [],
        "Fraud": ["AmountHigh", "NewDevice", "IPMismatch", "PastChargeback"],
    }

    # Priors (you can tune these)
    # Example: only ~10% of transactions are "high amount", ~20% are "new device", etc.
    tables = {
        "AmountHigh": [
            [[], [0.90, 0.10]],  # P(AmountHigh=F)=0.90, P(T)=0.10
        ],
        "NewDevice": [
            [[], [0.80, 0.20]],
        ],
        "IPMismatch": [
            [[], [0.85, 0.15]],
        ],
        "PastChargeback": [
            [[], [0.95, 0.05]],
        ],
        "Fraud": []
    }

    # Conditional table for Fraud:
    # Order of parent assignments MUST match parents["Fraud"] order:
    # [AmountHigh, NewDevice, IPMismatch, PastChargeback]
    #
    # Fraud probabilities list must match nodes["Fraud"] order: ["F", "T"]
    def fraud_row(a: str, d: str, ip: str, cb: str, p_fraud: float) -> List[Any]:
        # Store [P(F), P(T)]
        return [[a, d, ip, cb], [1.0 - p_fraud, p_fraud]]

    # Very simple risk logic (tunable):
    # - Base fraud probability is low
    # - Each risk signal pushes it higher
    base = 0.02

    def p(a: str, d: str, ip: str, cb: str) -> float:
        score = base
        if a == "T":
            score += 0.08
        if d == "T":
            score += 0.10
        if ip == "T":
            score += 0.12
        if cb == "T":
            score += 0.20
        # cap to avoid hitting 1.0
        return min(score, 0.95)

    for a in nodes["AmountHigh"]:
        for d in nodes["NewDevice"]:
            for ip in nodes["IPMismatch"]:
                for cb in nodes["PastChargeback"]:
                    tables["Fraud"].append(fraud_row(a, d, ip, cb, p(a, d, ip, cb)))

    return {"nodes": nodes, "parents": parents, "tables": tables}


def ensure_net_file(net_path: str) -> str:
    path = Path(net_path)
    if path.exists():
        return str(path)

    # Create the file if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build_default_fraud_net_json()
    path.write_text(json.dumps(data, indent=2))
    return str(path)


# -----------------------------
# 2) Load cases (or use defaults)
# -----------------------------

def load_cases(cases_path: Optional[str]) -> List[Dict[str, Any]]:
    """
    Expected JSON format:
    [
      {
        "case_id": "TXN-001",
        "amount_usd": 250.0,
        "evidence": {"AmountHigh": "F", "NewDevice": "T", "IPMismatch": "F", "PastChargeback": "F"}
      },
      ...
    ]
    """
    if cases_path is None:
        # Built-in sample cases (so this runs immediately)
        return [
            {
                "case_id": "TXN-001",
                "amount_usd": 45.00,
                "evidence": {"AmountHigh": "F", "NewDevice": "F", "IPMismatch": "F", "PastChargeback": "F"},
            },
            {
                "case_id": "TXN-002",
                "amount_usd": 980.00,
                "evidence": {"AmountHigh": "T", "NewDevice": "T", "IPMismatch": "F", "PastChargeback": "F"},
            },
            {
                "case_id": "TXN-003",
                "amount_usd": 120.00,
                "evidence": {"AmountHigh": "F", "NewDevice": "T", "IPMismatch": "T", "PastChargeback": "F"},
            },
            {
                "case_id": "TXN-004",
                "amount_usd": 2200.00,
                "evidence": {"AmountHigh": "T", "NewDevice": "T", "IPMismatch": "T", "PastChargeback": "T"},
            },
            {
                "case_id": "TXN-005",
                "amount_usd": 310.00,
                "evidence": {"AmountHigh": "F", "NewDevice": "F", "IPMismatch": "T", "PastChargeback": "T"},
            },
        ]

    raw = Path(cases_path).read_text()
    cases = json.loads(raw)
    if not isinstance(cases, list):
        raise ValueError("cases JSON must be a list of case objects")
    return cases


def validate_evidence(bn: BayesNet, evidence: Dict[str, str]) -> Dict[str, str]:
    """
    Keeps only valid evidence keys and values.
    This prevents typos from silently producing nonsense.
    """
    cleaned: Dict[str, str] = {}
    for var, val in evidence.items():
        if var not in bn.nodes:
            continue
        if val not in bn.nodes[var]:
            continue
        cleaned[var] = val
    return cleaned


# -----------------------------
# 3) Scoring + Prioritization
# -----------------------------

def fraud_probability(bn: BayesNet, evidence: Dict[str, str]) -> float:
    """
    Returns P(Fraud='T' | evidence).
    Uses enumerate_ask from your BayesNet implementation.
    """
    dist = bn.enumerate_ask("Fraud", evidence)
    return float(dist.get("T", 0.0))


def priority_score(p_fraud: float, amount_usd: Optional[float]) -> float:
    """
    Simple prioritization score:
    - risk is primary driver
    - amount adds a small multiplier to surface high-impact cases
    """
    if amount_usd is None:
        return p_fraud

    # Beginner-friendly: mild log scaling so $50 vs $5000 doesn't explode the score
    # log1p keeps it defined for amount=0
    import math
    impact = math.log1p(max(amount_usd, 0.0))  # ~3.9 for $50, ~8.5 for $5000
    return p_fraud * (1.0 + 0.15 * impact)


def prioritize_cases(
    bn: BayesNet,
    cases: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []

    for c in cases:
        case_id = str(c.get("case_id", "UNKNOWN"))
        amount = c.get("amount_usd", None)
        evidence = c.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}

        evidence = validate_evidence(bn, evidence)
        p = fraud_probability(bn, evidence)
        score = priority_score(p, amount)

        ranked.append(
            {
                "case_id": case_id,
                "amount_usd": amount,
                "p_fraud": round(p, 4),
                "priority_score": round(score, 4),
                "evidence": evidence,
            }
        )

    ranked.sort(key=lambda x: x["priority_score"], reverse=True)
    return ranked


def print_ranked(ranked: List[Dict[str, Any]], top: int = 10) -> None:
    print("\nFraud Review Priority List (highest first)\n")
    for i, r in enumerate(ranked[:top], start=1):
        print(f"{i}. {r['case_id']} | p_fraud={r['p_fraud']} | score={r['priority_score']} | amount={r['amount_usd']}")
        print(f"   evidence={r['evidence']}")
    print("")


# -----------------------------
# 4) CLI entrypoint
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fraud review prioritization using a Bayesian Network (TF/IDF-style ranking, but probabilistic)."
    )
    parser.add_argument(
        "--net",
        type=str,
        default="nets/fraud_review.json",
        help="Path to the Bayes Net JSON file. If it does not exist, a default one is created.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Path to cases JSON file. If omitted, built-in example cases are used.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many cases to display.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save ranked results as JSON.",
    )

    args = parser.parse_args()

    net_path = ensure_net_file(args.net)
    bn = BayesNet(net_path)

    cases = load_cases(args.cases)
    ranked = prioritize_cases(bn, cases)

    print_ranked(ranked, top=args.top)

    if args.output:
        Path(args.output).write_text(json.dumps(ranked, indent=2))
        print(f"Saved ranked cases to: {args.output}")


if __name__ == "__main__":
    main()
