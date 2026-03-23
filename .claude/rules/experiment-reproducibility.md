# Experiment Reproducibility

Every experiment must be reproducible on a different machine.

## Required in every experiment

1. **Set and record the random seed.** Call `numpy.random.seed(seed)` or pass seed to Config. Record the seed in the results JSON.

2. **Dump the full config.** Every parameter (n_bits, k_sparse, hidden, lr, wd, batch_size, max_epochs, seed) must appear in the results JSON. Do not rely on defaults being the same everywhere.

3. **Record the environment.** Include in results:
   - Python version (`sys.version`)
   - numpy version (`numpy.__version__`)
   - OS and CPU (`platform.platform()`, `platform.processor()`)
   - git commit hash (`git rev-parse HEAD`)

4. **Include baseline comparison.** Every experiment must state what it compares against and include the baseline numbers. "Better than SGD" means nothing without SGD's actual DMC/ARD/time.

5. **One variable at a time.** Change one thing from the baseline. If you change two things and the result improves, you don't know which one helped.

## In the experiment template

The template at `src/sparse_parity/experiments/_template.py` should include environment logging by default. If you're creating an experiment and the template doesn't log the environment, add it yourself.

## Verification

Before committing results, re-run with a different seed. If the result only works on one seed, say so in the findings doc.
