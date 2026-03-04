"""Loss functions, accuracy, and result reporting."""

import json
import time
from pathlib import Path


def hinge_loss(outs, ys):
    """Mean hinge loss: avg(max(0, 1 - out*y))."""
    return sum(max(0.0, 1.0 - o * y) for o, y in zip(outs, ys)) / len(ys)


def accuracy(outs, ys):
    """Fraction where sign(out) matches y."""
    correct = sum(1 for o, y in zip(outs, ys) if (1.0 if o >= 0 else -1.0) == y)
    return correct / len(ys)


def save_json(data, path):
    """Save dict as JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def save_markdown(content, path):
    """Save string as markdown file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)


def timestamp():
    """Generate a timestamp string for filenames."""
    return time.strftime('%Y%m%d_%H%M%S')
