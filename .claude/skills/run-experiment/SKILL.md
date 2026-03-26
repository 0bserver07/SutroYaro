---
name: run-experiment
description: Use when running a new experiment. Follows the two-phase protocol from LAB.md.
---

# Run Experiment

## Experiment types

**New method**: not in the registry (search_space.yaml). Create a new experiment file from the template. Add the method to the registry if it works.

**Existing method, new config**: method is in the registry but you're testing a different configuration (different n, k, hyperparameters). Use the existing experiment code or copy and modify.

Either way, the steps and output format are the same.

## Steps

1. Read DISCOVERIES.md. Check what's already proven. Do not repeat existing experiments.

2. Identify the hypothesis. Either from TODO.md, research/questions.yaml, or the user's request. State it as: "If we do X, then Y will happen because Z."

3. Create the experiment file. Copy `src/sparse_parity/experiments/_template.py`. Change one variable from the baseline.

4. Run the experiment. Capture results including accuracy, ARD, DMC, wall time. Record seed, config, environment (Python version, numpy version, OS, git hash).

5. Save Phase 1 output. Write `results/{exp_id}/results.json` with raw numbers, config, and environment. No interpretation in this file.

6. Verify. Re-run with a different seed. If the result only holds on one seed, note that.

7. Write Phase 2 findings. Create `docs/findings/{exp_id}.md` using the template from LAB.md. Use `Status: SUCCESS | PARTIAL | FAILED` (not "COMPLETED"). Reference the results JSON. Add analysis and impact.

8. Classify in research/log.jsonl. Use `"class": "WIN"` only if the result is a clear improvement. Use `"PARTIAL"` for mixed results. Use `"LOSS"` for negative results. All three are valid findings.

9. Update DISCOVERIES.md if the finding answers an open question or establishes a new fact.

## After merge: changelog and reporting

Not every experiment needs a changelog entry. After a PR is merged, the reviewing agent decides:

- **Add to changelog** if the result changes the best known method, maps a new frontier, answers an open question from DISCOVERIES.md, or is the first contribution from a new researcher.
- **Skip changelog** if the result confirms what's already known or is a minor null result.

The changelog entry goes in `docs/changelog.md` with the next version number, a short description of the finding, and a link to the findings doc. The reviewing agent writes this on merge, not the contributor.

If the result is significant enough for a meeting presentation, use the prepare-meeting skill to compile it into a report.

## Checklist

- [ ] DISCOVERIES.md read
- [ ] Hypothesis stated
- [ ] One variable changed from baseline
- [ ] Experiment run with seed recorded
- [ ] results.json saved with config + environment
- [ ] Verified with different seed
- [ ] Findings doc written in docs/findings/ with Status: SUCCESS/PARTIAL/FAILED
- [ ] log.jsonl updated with correct class (WIN/PARTIAL/LOSS)
- [ ] DISCOVERIES.md updated if applicable
