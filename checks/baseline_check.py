#!/usr/bin/env python3
"""
Re-establish baselines on this machine. Run at the start of each session.

Verifies that core methods produce expected results on this hardware.
Prints baseline numbers that all experiments should compare against.

Usage:
    PYTHONPATH=src python3 checks/baseline_check.py
"""

import sys

try:
    from harness import measure_sparse_parity
except ImportError:
    print("ERROR: Cannot import harness. Run with PYTHONPATH=src")
    sys.exit(1)

baselines = [
    {"method": "gf2", "n_bits": 20, "k_sparse": 3, "expected_accuracy": 1.0,
     "label": "GF(2) n=20/k=3"},
    {"method": "sgd", "n_bits": 20, "k_sparse": 3, "hidden": 200, "lr": 0.1,
     "max_epochs": 200, "n_train": 1000, "expected_accuracy": 0.95,
     "label": "SGD n=20/k=3"},
    {"method": "km", "n_bits": 20, "k_sparse": 3, "n_train": 100,
     "expected_accuracy": 1.0, "label": "KM n=20/k=3"},
]

print("Establishing baselines on this machine...\n")
print(f"{'Method':<20} {'Acc':>6} {'ARD':>10} {'DMC':>10} {'Time':>10} {'Status'}")
print(f"{'-'*20} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

all_pass = True
results = []

for b in baselines:
    label = b.pop("label")
    expected = b.pop("expected_accuracy")

    result = measure_sparse_parity(**b)
    acc = result.get("accuracy", 0)
    ard = result.get("ard", "n/a")
    dmc = result.get("dmc", "n/a")
    t = result.get("time_s", "n/a")

    ok = acc >= expected
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False

    ard_str = f"{ard:>10,.0f}" if isinstance(ard, (int, float)) else f"{ard:>10}"
    dmc_str = f"{dmc:>10,.0f}" if isinstance(dmc, (int, float)) else f"{dmc:>10}"
    t_str = f"{t:>10.4f}" if isinstance(t, (int, float)) else f"{t:>10}"

    print(f"{label:<20} {acc:>6.2f} {ard_str} {dmc_str} {t_str} {status}")
    results.append({"label": label, "result": result, "ok": ok})

print()
if all_pass:
    print("All baselines verified. Ready to experiment.")
else:
    print("WARNING: Some baselines failed. Check environment before proceeding.")
    sys.exit(1)
