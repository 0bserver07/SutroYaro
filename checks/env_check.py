#!/usr/bin/env python3
"""
Pre-flight environment check. Run before any agent session.

Usage:
    PYTHONPATH=src python3 checks/env_check.py
"""

import sys
import json

results = {"pass": True, "checks": []}


def check(name, fn):
    try:
        ok, detail = fn()
        results["checks"].append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            results["pass"] = False
    except Exception as e:
        results["checks"].append({"name": name, "ok": False, "detail": str(e)})
        results["pass"] = False


def check_python_version():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    return ok, f"Python {v.major}.{v.minor}.{v.micro}"


def check_numpy():
    try:
        import numpy as np
        return True, f"numpy {np.__version__}"
    except ImportError:
        return False, "numpy not installed"


def check_config_import():
    try:
        from sparse_parity.config import Config
        c = Config()
        return True, f"Config imports, default n_bits={c.n_bits}"
    except Exception as e:
        return False, str(e)


def check_tracker_import():
    try:
        from sparse_parity.tracker import MemTracker
        t = MemTracker()
        t.write("test", 10)
        t.read("test")
        s = t.summary()
        return True, f"MemTracker works, ARD={s['weighted_ard']}"
    except Exception as e:
        return False, str(e)


def check_data_generation():
    try:
        from sparse_parity.config import Config
        from sparse_parity.fast import generate
        c = Config(n_bits=20, k_sparse=3, n_train=100, n_test=50, seed=42)
        x_tr, y_tr, x_te, y_te, secret = generate(c)
        ok = x_tr.shape == (100, 20) and len(secret) == 3
        return ok, f"Data shape {x_tr.shape}, secret={secret}"
    except Exception as e:
        return False, str(e)


def check_harness():
    try:
        from harness import measure_sparse_parity
        result = measure_sparse_parity("gf2", n_bits=10, k_sparse=3, seed=42)
        ok = result.get("accuracy", 0) >= 0.99
        return ok, f"Harness works, GF2 accuracy={result.get('accuracy')}"
    except Exception as e:
        return False, str(e)


check("python_version", check_python_version)
check("numpy", check_numpy)
check("config_import", check_config_import)
check("tracker_import", check_tracker_import)
check("data_generation", check_data_generation)
check("harness", check_harness)

# Print results
for c in results["checks"]:
    status = "PASS" if c["ok"] else "FAIL"
    print(f"  [{status}] {c['name']}: {c['detail']}")

if results["pass"]:
    print("\nAll checks passed.")
else:
    print("\nSome checks FAILED. Fix before running experiments.")
    sys.exit(1)
