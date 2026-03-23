---
name: run-experiment
description: Use when running a new experiment. Follows the two-phase protocol from LAB.md.
---

# Run Experiment

## Steps

1. Read DISCOVERIES.md. Check what's already proven. Do not repeat existing experiments.

2. Identify the hypothesis. Either from TODO.md, research/questions.yaml, or the user's request. State it as: "If we do X, then Y will happen because Z."

3. Create the experiment file. Copy `src/sparse_parity/experiments/_template.py`. Change one variable from the baseline.

4. Run the experiment. Capture results including accuracy, ARD, DMC, wall time. Record seed, config, environment (Python version, numpy version, OS, git hash).

5. Save Phase 1 output. Write `results/{exp_id}/results.json` with raw numbers, config, and environment. No interpretation in this file.

6. Verify. Re-run with a different seed. If the result only holds on one seed, note that.

7. Write Phase 2 findings. Create `docs/findings/{exp_id}.md` using the template from LAB.md. Reference the results JSON. Add analysis and impact.

8. Update DISCOVERIES.md if the finding answers an open question or establishes a new fact.

9. Add to research/log.jsonl.

## Checklist

- [ ] DISCOVERIES.md read
- [ ] Hypothesis stated
- [ ] One variable changed from baseline
- [ ] Experiment run with seed recorded
- [ ] results.json saved with config + environment
- [ ] Verified with different seed
- [ ] Findings doc written in docs/findings/
- [ ] DISCOVERIES.md updated if applicable
- [ ] log.jsonl updated
