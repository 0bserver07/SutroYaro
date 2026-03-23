---
name: prepare-meeting
description: Use before a Sutro Group meeting to compile results and prepare a presentation report.
---

# Prepare Meeting

## Steps

1. Read the latest catch-up in `docs/catchups/` for what happened this week.

2. Check which experiments were run since the last meeting. Look at `research/log.jsonl` and `docs/findings/` for new entries.

3. Compile results into a report. Create `docs/catchups/meeting-{N}-report.md` with:
   - What experiments were run and their results
   - Tables with actual numbers (DMC, ARD, accuracy, time)
   - What changed in the codebase (new features, infrastructure)
   - Open questions for discussion
   - Links to findings docs and plots

4. Check for plots in `results/plots/`. Reference them in the report if relevant.

5. List open GitHub issues that are relevant to discuss.

6. Add the report to mkdocs.yml nav under Weekly Catch-Up.

7. Run the anti-slop skill on the report before finalizing.

## Checklist

- [ ] Latest catch-up read
- [ ] New experiments identified
- [ ] Report created with tables and numbers
- [ ] Plots referenced
- [ ] Open issues listed
- [ ] Nav updated
- [ ] Anti-slop pass done
