"""Compute backends for running experiments.

Abstracts where and how the harness runs:
- LocalBackend: direct Python import (current default behavior)
- ModalBackend: GPU execution via Modal Labs (prototype)
- RemoteBackend: HTTP API call to a hosted harness (prototype)
"""

import json
import signal
from abc import ABC, abstractmethod


class _HarnessTimeout(Exception):
    """Raised when a harness call exceeds its time budget."""


def _timeout_handler(signum, frame):
    raise _HarnessTimeout("Harness call timed out")


class HarnessBackend(ABC):
    """Interface for running experiments."""

    @abstractmethod
    def run(self, challenge, method, n_bits=20, k_sparse=3, **kwargs) -> dict:
        """Run one experiment. Returns dict with accuracy, ard, dmc, time_s, etc."""
        pass


class LocalBackend(HarnessBackend):
    """Run harness locally via Python import. Current default behavior."""

    def __init__(self, timeout=10.0):
        self.timeout = timeout

    def run(self, challenge, method, n_bits=20, k_sparse=3, **kwargs):
        from sparse_parity.eval.registry import get_harness_fn

        measure_fn = get_harness_fn(challenge)

        try:
            # Set timeout (Unix only, graceful)
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(self.timeout))

            result = measure_fn(
                method=method,
                n_bits=n_bits,
                k_sparse=k_sparse,
                **kwargs,
            )

            signal.alarm(0)  # cancel alarm
            signal.signal(signal.SIGALRM, old_handler)

            # If the harness returned an error dict, treat as failure
            if "error" in result and result.get("accuracy") is None:
                result.setdefault("accuracy", 0.0)

            return result

        except _HarnessTimeout:
            signal.alarm(0)
            return {
                "accuracy": 0.0,
                "ard": None,
                "dmc": None,
                "time_s": self.timeout,
                "total_floats": None,
                "error": f"Method '{method}' timed out after {self.timeout}s",
                "method": method,
            }
        except Exception as e:
            signal.alarm(0)
            return {
                "accuracy": 0.0,
                "ard": None,
                "dmc": None,
                "time_s": None,
                "total_floats": None,
                "error": f"Method '{method}' raised: {type(e).__name__}: {e}",
                "method": method,
            }


class ModalBackend(HarnessBackend):
    """Run experiments on Modal Labs GPU.

    Requires: pip install modal, MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars.

    Usage:
        backend = ModalBackend(gpu="L4")
        result = backend.run("sparse-parity", "sgd", n_bits=20, k_sparse=3)
    """

    def __init__(self, gpu="L4"):
        self.gpu = gpu
        # Don't import modal at class level -- it's an optional dep

    def run(self, challenge, method, n_bits=20, k_sparse=3, **kwargs):
        try:
            import modal  # noqa: F401
        except ImportError:
            return {"error": "modal not installed. Run: pip install modal", "accuracy": 0.0}

        # Define the Modal function that runs the harness remotely.
        # This is a prototype -- the actual Modal deployment needs:
        # 1. A Modal app with the harness code
        # 2. GPU selection (L4, A100)
        # 3. Result serialization
        # For now, return a clear error explaining what's needed.
        return {
            "error": f"Modal backend not yet deployed. GPU={self.gpu}. "
                     f"See bin/gpu_energy.py for existing Modal integration.",
            "accuracy": 0.0,
        }


class RemoteBackend(HarnessBackend):
    """Run experiments via HTTP API call to a hosted harness.

    Usage:
        backend = RemoteBackend("https://harness.example.com/run")
        result = backend.run("sparse-parity", "sgd")
    """

    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url

    def run(self, challenge, method, n_bits=20, k_sparse=3, **kwargs):
        import urllib.request

        payload = json.dumps({
            "challenge": challenge,
            "method": method,
            "n_bits": n_bits,
            "k_sparse": k_sparse,
            **kwargs,
        }).encode()

        try:
            req = urllib.request.Request(
                self.endpoint_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"error": str(e), "accuracy": 0.0}


def get_backend(name="local", **kwargs):
    """Factory function for backends.

    Args:
        name: "local", "modal", or an HTTP URL for RemoteBackend.
        **kwargs: Passed to the backend constructor.

    Returns:
        HarnessBackend instance.
    """
    if name == "local":
        return LocalBackend(**kwargs)
    elif name == "modal":
        return ModalBackend(**kwargs)
    elif name.startswith("http"):
        return RemoteBackend(endpoint_url=name, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}. Use 'local', 'modal', or an HTTP URL.")
