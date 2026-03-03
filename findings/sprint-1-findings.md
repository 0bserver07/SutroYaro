# Sprint 1 Findings (02 Mar 2026)

## Summary
2.5 hours of work. Automatic identification of gradient fusion strategy via Claude. Cache reuse improved by 16%.

## Setup
- 3-bit parity task, tiny neural network
- Pure Python implementation (no PyTorch overhead)
- <1 second total runtime constraint

## Key Finding: ARD Bottleneck in Backprop

W1 alone accounts for 6,000 of 19,013 total floats read (32%), with reuse distance ~15,000 unchanged by fusion. Similarly b1 (2,000 floats, 11%) unchanged.

The buffers improved by fusion (dW2: 16,005 -> 3,002, db2: 18,005 -> 5,002) only contribute 1,001 floats out of 19,013 total -- just 5% of the weighted sum.

## Conclusion
Gradient fusion fixes easy wins (gradient buffers), but the real bottleneck is parameter tensors read twice with the entire forward+backward pass in between. Fundamentally different algorithm needed.

## Candidate Next Steps
- Per-layer forward-backward (changes math)
- Forward-Forward algorithm
- Local learning rules

## Artifacts
- Code: https://github.com/cybertronai/sutro
- Sparse parity benchmark: sparse_parity_benchmark.py
- Gemini brainstorming sessions on ARD
