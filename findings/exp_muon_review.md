# Muon Optimizer — Literature Review

## Hypothesis
If we substitute standard optimizers with Muon's Newton-Schulz matrix orthogonalization for the sparse parity task, we expect it to *worsen* energy efficiency (increase ByteDMD) compared to SGD. Newton-Schulz iterations require multiple heavy matrix multiplications per step, generating high byte-level data movement through intermediate matrices. Furthermore, its overhead scales inversely with batch size, making it extremely inefficient for our single-sample or small-batch regimes.

## Key Facts from Paper
- **Algorithm:** Muon orthogonalizes the update matrix from SGD-momentum using a Newton-Schulz (NS) iteration, replacing the update with the nearest semi-orthogonal matrix ($UV^\top$ from SVD).
- **Scope:** It applies only to 2D hidden layer parameters (e.g., $W_1$). Embedding, bias, and output parameters must still be optimized by AdamW/standard methods.
- **Memory footprint:** Muon uses SGD-momentum (one buffer), substituting Adam's second moment accumulator with NS iteration matrix temporaries ($A, B, X$) computed in bfloat16.
- **Computational overhead:** The overhead is bounded by $Tm/B$ where $T \approx 5$ is NS steps, $m$ is model dimension, and $B$ is batch size in tokens. For large transformers ($B > 500,000$), FLOP overhead is < 1%.
- **Performance:** Set speedrun records for training NanoGPT and CIFAR-10. It reduced the training time of a 1.5B parameter transformer on HellaSwag from 13.3 to 10 hours.
- **Limitations:** Muon has a slower per-step wallclock time than standard optimizers; its success hinges on reducing the overall number of training steps (epochs) significantly.

## Relevance to Sutro Group
- **Energy efficiency (ByteDMD):** Muon is highly detrimental to ByteDMD. The 5-step NS iteration performs operations like $A = XX^\top$ and $B = bA + cA^2$, aggressively reading and writing matrices. Each matmul natively incurs a massive ByteDMD cost (as intermediate values flush the LRU stack), ruining the cache locality that makes single-sample SGD efficient.
- **Small batch scaling:** Our sparse parity setups use small networks ($m=200$ to $1000$) with tiny batch sizes ($B=1$ to $32$). The FLOP overhead $Tm/B$ becomes astronomical, e.g., $5 \times 1000 / 32 \approx 156\times$ the baseline FLOPs.
- **Problem structure:** Sparse parity's difficulty is primarily searching for exactly $k$ secret bits (a combinatorial feature extraction problem), not resolving a poorly conditioned loss landscape in hidden layer activations, which is what orthogonalization fixes. 

## Comparison to Our Methods
- **SGD:** Single-sample SGD has excellent L1-cache locality yielding low ByteDMD/ARD. Muon would require accumulating a batch update and running an $O(m^3)$ matrix orthogonalization, destroying our $100\%$ L1 hit rates and resulting in a much higher DMC.
- **EGD (Egalitarian Gradient Descent):** Existing experiments (`exp_egd`) show that gradient SVD-normalization cuts epochs by ~2x but its per-step SVD overhead makes wall time 12% worse. Muon's NS iteration is an approximation to SVD, meaning it would run into the same exact bottleneck for this task: optimizing the step count at the expense of per-step computational & data movement overhead.
- **GF(2) & KM-min:** These perform extremely well on parity (DMC ~3500-8600, ~1ms wall-clock). Muon, as a gradient-based iterative optimizer, fundamentally has no path to beating $O(n)$ or deterministic matrix algebra solvers on purely logical tasks.

## Open Questions
- Does Newton-Schulz iteration yield *any* net epoch reduction for 20-bit sparse parity, or does orthogonalizing the gradient matrices corrupt the needle-in-a-haystack subsets of weights that are actually learning?
- Can the Newton-Schulz step be computed iteratively in a block-tiled manner to fit completely within the L1 cache, thereby bounding its otherwise catastrophic ByteDMD cost?
- What is the empirical ByteDMD cost of running 1 step of Muon vs. 1 step of SGD on a $1000 \times 20$ matrix using `bytedmd`'s tracker?

## References
- Primary Source: [Muon Blog Post](https://kellerjordan.github.io/posts/muon/)
- Code: [KellerJordan/Muon GitHub](https://github.com/KellerJordan/Muon)
- Metric: [ByteDMD Metric Definition](https://github.com/cybertronai/ByteDMD)
- Internal Context: `DISCOVERIES.md` (`exp_egd` and scaling properties)
