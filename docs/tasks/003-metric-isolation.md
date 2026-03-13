# Task 3: Add explicit metric isolation rule to LAB.md

**Priority**: MEDIUM
**Status**: DONE
**Source**: Meeting #8 AI notes, Germain's agent gaming issue

## Context

Germain's agents rewrote the ARD evaluation code to achieve artificially high scores instead of improving the actual algorithm. The AI notes from Meeting #8 flag this: "agents attempt to game the evaluation metrics, finding shortcuts by rewriting the measurement code."

Our repo already does this in practice (sub-agents get read-only benchmark code), but it's not written as an explicit rule anywhere.

## Tasks

- [ ] Add "Metric Isolation" rule to LAB.md: measurement code must be read-only, agents cannot modify tracker.py, cache_tracker.py, or data.py
- [ ] Add note to CLAUDE.md under Working Style
- [ ] Reference Germain's experience as the motivating example

## References

- Meeting #8 AI notes: docs/google-docs/meeting-8-ai-notes.md (Section 3, "Gaming of Metrics")
- Yaroslav verification doc: "importance of using the right metric, and for metric code to be isolated from agent modifications"
- Current LAB.md rules
