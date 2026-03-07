# Experiment exp_decision_tree: Decision Trees for Sparse Parity

**Date**: 2026-03-06
**Status**: FAILED (no model achieves 100% test accuracy)
**Answers**: Tree-based methods cannot reliably solve sparse parity due to greedy splitting

## Hypothesis

A depth-k binary decision tree can learn k-parity exactly if each split targets one of the secret bits. Random forests with max_depth=k would implicitly search the subset space. ExtraTrees with random splits might bypass the greedy information-gain trap. RISK: greedy splitting by information gain fails for parity because individual bits have zero marginal correlation with the label.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20, 50 |
| k_sparse | 3, 5 |
| models | DecisionTree, RandomForest (100 trees), ExtraTrees (500 trees) |
| max_depth | k, 2*k, unlimited |
| n_train | 5000 (k=3), 10000 (k=5) |
| n_test | 1000 |
| seeds | 42, 43, 44 |

## Results

### Test accuracy (averaged over 3 seeds)

| Model | n=20/k=3 | n=50/k=3 | n=20/k=5 |
|-------|----------|----------|----------|
| DT_depth_k | 50.0% | 51.0% | 49.8% |
| DT_depth_2k | 50.4% | 50.0% | 52.0% |
| DT_unlimited | 69.4% | 55.0% | 56.6% |
| RF_depth_k | 55.4% | 50.6% | 48.3% |
| RF_depth_2k | 79.7% | 55.4% | 60.8% |
| RF_depth_unlimited | 91.8% | 61.4% | 66.1% |
| ET_depth_k | 50.8% | 51.5% | 48.7% |
| ET_depth_2k | 79.2% | 56.7% | 60.4% |
| ET_depth_unlimited | 92.5% | 65.6% | 67.1% |

### Timing (average seconds)

| Model | n=20/k=3 | n=50/k=3 | n=20/k=5 |
|-------|----------|----------|----------|
| DT_depth_k | 0.004s | 0.007s | 0.020s |
| DT_depth_2k | 0.005s | 0.017s | 0.016s |
| DT_unlimited | 0.009s | 0.024s | 0.023s |
| RF_depth_k | 0.137s | 0.104s | 0.113s |
| RF_depth_2k | 0.092s | 0.108s | 0.136s |
| RF_depth_unlimited | 0.102s | 0.129s | 0.168s |
| ET_depth_k | 0.325s | 0.407s | 0.504s |
| ET_depth_2k | 0.348s | 0.492s | 0.594s |
| ET_depth_unlimited | 0.422s | 0.621s | 0.763s |

### Comparison with baselines

| Method | n=20/k=3 | n=50/k=3 | n=20/k=5 |
|--------|----------|----------|----------|
| Best tree (ET unlimited) | 92.5% | 65.6% | 67.1% |
| SGD (baseline) | 100% | 54% (FAIL) | 100% |
| Random search | 100% | 100% | 100% |
| Fourier exhaustive | 100% | 100% | 100% |

## Analysis

### What worked

- **ExtraTrees unlimited depth is the best tree method**, reaching 92.5% on n=20/k=3 (the easiest config). Random splits help ExtraTrees explore more diverse feature combinations than greedy trees.
- **Ensembles substantially outperform single trees**: RF_unlimited (91.8%) vs DT_unlimited (69.4%) on n=20/k=3. Averaging many overfitting trees produces a better approximation.
- **Depth 2*k helps significantly over depth k**: RF_depth_2k gets 79.7% vs RF_depth_k at 55.4% on n=20/k=3. The tree needs extra depth beyond k to route through irrelevant features.
- **Feature importances do recover the secret bits** in unlimited-depth models, but only because the tree memorizes training data patterns rather than learning the parity function.
- **All models achieve 100% train accuracy** with unlimited depth, confirming trees can memorize parity. The problem is generalization.

### What didn't work

- **No tree model achieves 100% test accuracy on any config.** The best result is 92.5% (ET_unlimited on n=20/k=3). This confirms the core hypothesis risk: greedy splitting cannot learn parity.
- **Depth-k trees perform at chance (~50%)** on all configs. Despite k being the theoretical minimum depth to represent k-parity, greedy information-gain splitting never finds the right split sequence because individual bits have zero correlation with the label.
- **n=50/k=3 is hardest for trees** (best: 65.6%) even though C(50,3)=19,600 is moderate. More irrelevant features mean more distractors for the greedy splitter. This is the same config where SGD also fails (54%).
- **n=20/k=5 is similarly poor** (best: 67.1%). Higher parity order makes the interaction harder to capture with axis-aligned splits.
- **ExtraTrees' random splits don't solve the problem** despite being the best variant. 500 trees with random splits still cannot reliably discover the k-way interaction.

### Surprise

- **Trees beat SGD on n=50/k=3**: ET_unlimited gets 65.6% vs SGD's 54%. Both fail, but the ensemble of random trees captures more of the parity signal than a neural net with vanilla SGD. Neither comes close to the combinatorial methods (random search, Fourier) which get 100%.
- **The gap between train and test accuracy is enormous**: 100% train vs 55-67% test for unlimited trees on harder configs. This is extreme overfitting -- the tree memorizes n_train parity values but cannot generalize the function.
- **Depth-k trees are functionally random guessing** (50% accuracy), even though a depth-k tree CAN represent k-parity. The gap between representational capacity and learnability via greedy splitting is total.

## Open Questions

- Would boosted trees (XGBoost, LightGBM) do better by iteratively correcting residuals?
- Could oblique decision trees (splits on linear combinations of features) learn parity?
- What if we gave the tree product features (x_i * x_j) as inputs -- would depth-1 trees solve 2-parity?
- How many training samples would an unlimited-depth random forest need to approach 100% test accuracy?

## Files

- Experiment: `src/sparse_parity/experiments/exp_decision_tree.py`
- Results: `results/exp_decision_tree/results.json`
