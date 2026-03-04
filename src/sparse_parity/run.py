"""Main runner: execute all phases sequentially, produce JSON + markdown + plots."""

import copy
import json
import time
from pathlib import Path

from .config import Config, SCALE_CONFIG
from .data import generate
from .model import init_params
from .train import train
from .train_fused import train_fused
from .train_perlayer import train_perlayer
from .metrics import save_json, save_markdown, timestamp


RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'


def run_experiment(config, label=''):
    """Run all 3 training variants on same data, return comparison."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {label} (n={config.n_bits}, k={config.k_sparse})")
    print(f"{'='*70}")

    data = generate(config)
    x_train, y_train, x_test, y_test, secret = data
    print(f"  Secret indices: {secret}")
    print(f"  Params: {config.total_params:,}")

    results = {}

    # Phase 2: Standard backprop
    print(f"\n  [Phase 2] Standard backprop...")
    W1, b1, W2, b2 = init_params(config)
    r = train(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config)
    print(f"    Accuracy: {r['best_test_acc']:.0%} in {r['elapsed_s']:.3f}s")
    print(f"    ARD: {r['tracker']['weighted_ard']:,.0f}" if r['tracker'] else "    ARD: N/A")
    results['standard'] = r

    # Phase 4a: Fused
    print(f"\n  [Phase 4a] Fused layer-wise...")
    W1, b1, W2, b2 = init_params(config)
    r = train_fused(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config)
    print(f"    Accuracy: {r['best_test_acc']:.0%} in {r['elapsed_s']:.3f}s")
    print(f"    ARD: {r['tracker']['weighted_ard']:,.0f}" if r['tracker'] else "    ARD: N/A")
    results['fused'] = r

    # Phase 4b: Per-layer
    print(f"\n  [Phase 4b] Per-layer forward-backward...")
    W1, b1, W2, b2 = init_params(config)
    r = train_perlayer(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config)
    print(f"    Accuracy: {r['best_test_acc']:.0%} in {r['elapsed_s']:.3f}s")
    print(f"    ARD: {r['tracker']['weighted_ard']:,.0f}" if r['tracker'] else "    ARD: N/A")
    results['perlayer'] = r

    return results, secret


def generate_report(all_results, ts):
    """Generate markdown comparison report."""
    lines = [
        f"# Sparse Parity Experiment Results",
        f"",
        f"**Generated**: {ts}",
        f"",
    ]

    for label, (results, secret) in all_results.items():
        lines.append(f"## {label}")
        lines.append(f"")
        lines.append(f"Secret indices: {secret}")
        lines.append(f"")
        lines.append(f"| Method | Best Accuracy | ARD (weighted) | Time |")
        lines.append(f"|--------|--------------|----------------|------|")

        for method, r in results.items():
            acc = f"{r['best_test_acc']:.0%}"
            ard = f"{r['tracker']['weighted_ard']:,.0f}" if r.get('tracker') else "N/A"
            t = f"{r['elapsed_s']:.3f}s"
            lines.append(f"| {method} | {acc} | {ard} | {t} |")

        lines.append(f"")

        # ARD comparison
        if all(r.get('tracker') for r in results.values()):
            std_ard = results['standard']['tracker']['weighted_ard']
            for method in ['fused', 'perlayer']:
                if method in results and results[method].get('tracker'):
                    m_ard = results[method]['tracker']['weighted_ard']
                    pct = (1 - m_ard / std_ard) * 100 if std_ard > 0 else 0
                    lines.append(f"**{method}** ARD improvement over standard: **{pct:.1f}%**")
            lines.append(f"")

    return '\n'.join(lines)


def try_plot(all_results, ts):
    """Generate plots if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [PLOT] matplotlib not available, skipping plots")
        return

    for label, (results, _) in all_results.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss curves
        for method, r in results.items():
            axes[0].plot(r['train_losses'], label=f'{method} train', alpha=0.7)
            axes[0].plot(r['test_losses'], label=f'{method} test', linestyle='--', alpha=0.7)
        axes[0].set(xlabel='Epoch', ylabel='Hinge Loss', title=f'{label} - Loss')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Accuracy curves
        for method, r in results.items():
            axes[1].plot(r['test_accs'], label=method)
        axes[1].set(xlabel='Epoch', ylabel='Test Accuracy', title=f'{label} - Accuracy')
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # ARD comparison bar chart
        methods = []
        ards = []
        for method, r in results.items():
            if r.get('tracker'):
                methods.append(method)
                ards.append(r['tracker']['weighted_ard'])
        if methods:
            bars = axes[2].bar(methods, ards, color=['#2196F3', '#FF9800', '#4CAF50'])
            axes[2].set(ylabel='Weighted ARD (floats)', title=f'{label} - ARD Comparison')
            axes[2].grid(True, alpha=0.3, axis='y')
            for bar, v in zip(bars, ards):
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{v:,.0f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle(f'Sparse Parity: {label}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = RESULTS_DIR / f'{ts}_{label.lower().replace(" ", "_")}_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  [PLOT] Saved: {plot_path.name}")
        plt.close(fig)


def main():
    """Run the full pipeline: 3-bit baseline + 20-bit scaling."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = timestamp()
    total_start = time.time()

    all_results = {}

    # Phase 1-4: 3-bit parity
    config_3bit = Config()
    results_3bit, secret_3bit = run_experiment(config_3bit, '3-bit parity')
    all_results['3-bit parity'] = (results_3bit, secret_3bit)

    # Phase 5: Scale to 20-bit
    results_20bit, secret_20bit = run_experiment(SCALE_CONFIG, '20-bit sparse parity')
    all_results['20-bit sparse parity'] = (results_20bit, secret_20bit)

    # Save JSON
    json_path = RESULTS_DIR / f'{ts}_results.json'
    json_data = {}
    for label, (results, secret) in all_results.items():
        json_data[label] = {
            'secret': secret,
            'methods': {m: {k: v for k, v in r.items() if k != 'tracker'}
                        for m, r in results.items()},
            'ard': {m: r['tracker'] for m, r in results.items() if r.get('tracker')},
        }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\n  [JSON] Saved: {json_path.name}")

    # Save markdown report
    report = generate_report(all_results, ts)
    md_path = RESULTS_DIR / f'{ts}_report.md'
    with open(md_path, 'w') as f:
        f.write(report)
    print(f"  [MD] Saved: {md_path.name}")

    # Generate plots
    try_plot(all_results, ts)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  DONE in {total_elapsed:.2f}s")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
