# Experiment E: Forward-Forward Algorithm for Sparse Parity

**Date**: 2026-03-04
**Status**: Complete (partial -- 3-bit solved, 20-bit did not converge)

## Summary

Implemented Hinton's Forward-Forward (FF) algorithm in pure Python and tested it on sparse parity. FF solves 3-bit parity (n=3, k=3) perfectly in 4 epochs, but:
1. **Does NOT converge on 20-bit** parity (n=20, k=3) within reasonable time
2. **Has 25x HIGHER ARD than backprop**, not lower -- the opposite of our hypothesis

## Key Results

| Method             | n_bits | Test Acc | Weighted ARD | Reads | Writes |
|--------------------|--------|----------|--------------|-------|--------|
| Backprop           | 3      | 100%     | 447          | 9     | 9      |
| Forward-Forward    | 3      | 100%     | 11,303       | 14    | 18     |
| Backprop           | 20     | 96.5%    | 10,229       | 9     | 9      |
| Forward-Forward    | 20     | 58.5%    | 277,256      | 14    | 18     |

**ARD ratio**: Backprop has ~25x lower ARD than Forward-Forward per training step.

## Why FF Has Higher ARD (Counterintuitive)

The initial hypothesis was that FF would have smaller ARD because there is no backward pass -- parameters are only accessed "locally." However:

1. **FF does TWO forward passes per sample** (positive + negative), each requiring a full read of all layer weights. Backprop does one forward + one backward, but the backward pass reuses the same weights that were just read in the forward pass (shorter reuse distance).

2. **Each layer's weights are read 4 times in FF**: once for positive forward, once for positive update, once for negative forward, once for negative update. In backprop, W1 is read once in forward and once in backward -- and the backward read has a much shorter reuse distance because it happens shortly after the forward read.

3. **The "local" advantage is illusory for small networks**: In a 2-layer MLP, backprop's gradient chain is very short. The locality advantage of FF would only manifest in very deep networks (10+ layers) where backprop must store many intermediate activations and the backward pass touches buffers that were written long ago.

4. **FF has more writes**: 18 writes vs 9 for backprop. Each positive/negative pass writes separate activation buffers and updates weights twice.

## Forward-Forward Implementation Details

- **Label embedding**: Label (+1 or -1) prepended as first input dimension
- **Goodness function**: sum of squared ReLU activations per layer
- **Objective**: Positive data goodness > threshold; negative data goodness < threshold
- **Architecture**: 2-layer MLP with normalization between layers (Hinton's recommendation)
- **Threshold**: 2.0 (tuned)
- **Learning rate**: 0.01

## 20-bit Parity: Why FF Fails

FF does not solve 20-bit sparse parity (best: 58.5% after 2 epochs before timeout). Reasons:

1. **Greedy layer-wise learning**: Each layer optimizes its own goodness independently. Sparse parity requires learning XOR of 3 out of 20 bits -- this requires coordinated multi-layer feature extraction that greedy layer-wise objectives struggle with.

2. **Weak gradient signal**: The FF gradient for goodness is 2*h_j per neuron. For sparse parity, the signal from the 3 relevant bits is overwhelmed by the 17 noise bits. Backprop can propagate a focused error signal through both layers.

3. **Runtime**: Pure Python FF is slow -- 43 seconds per epoch with n_train=500, hidden=500. Each epoch requires 2x the forward computations of backprop (positive + negative passes).

## Implications for Sutro Group

**FF is NOT a good fit for energy-efficient sparse parity training** because:

- Higher ARD per step (25x worse than backprop)
- Cannot solve the hard 20-bit task
- More total computation (2 forward passes per sample)

**Where FF COULD help**:
- Very deep networks (10+ layers) where backprop's activation storage creates large reuse distances
- Tasks where negative data is easy to generate (images, not parity)
- Hardware with limited memory where avoiding the backward pass reduces peak memory

**Better directions for energy efficiency on sparse parity**:
- Fused forward+backward (already shown in Exp A to reduce ARD)
- Per-layer training with backprop (local gradients, but still uses chain rule)
- Smaller hidden sizes with curriculum learning

## Files

- Experiment: `src/sparse_parity/experiments/exp_e_forward_forward.py`
- Results: `results/exp_e_forward_forward/results.json`
