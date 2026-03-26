---
name: info-defrag
description: Use periodically (weekly or before a release) to find stale numbers, outdated descriptions, broken links, and inconsistencies across the codebase. Run after merging multiple PRs or before preparing a meeting report.
---

# Info Defrag

Scan the codebase for stale information, inconsistencies, and broken references. This catches drift that happens when experiments get merged but docs don't get updated.

## What to check

### 1. Experiment count

The number of experiments appears in multiple files. They drift.

```bash
# Actual count
wc -l research/log.jsonl
grep -c "^| exp" DISCOVERIES.md

# Where the number appears
grep -rn "33 experiment\|34 experiment\|35 experiment\|36 experiment" docs/ CLAUDE.md README.md
```

Fix any file that shows a different number than the log.

### 2. Best methods table

CLAUDE.md and DISCOVERIES.md both have a methods ranking table. They should match.

```bash
# Compare the two
grep -A10 "Current Best Methods" CLAUDE.md
grep -A10 "DMC baseline rankings" DISCOVERIES.md
```

### 3. People descriptions

Contributors change roles. Check docs/context.md People table against recent git log.

```bash
# Who contributed recently
git log --format="%an" --since="2 weeks ago" | sort -u
```

If someone submitted experiments, their bio should mention it.

### 4. Timeline coverage

The gantt chart in docs/context.md should cover recent work.

```bash
# Last date in the timeline
grep -o "2026-[0-9-]*" docs/context.md | sort | tail -1

# Last commit date
git log --format="%ai" -1 | cut -d' ' -f1
```

If the gap is more than a week, extend the timeline.

### 5. Index pages

Check that index files list all their children.

```bash
# Catchups
ls docs/catchups/*.md | wc -l
grep -c "\.md" docs/catchups/index.md

# Sessions
ls docs/sessions/*.md | grep -v transcript | wc -l
grep -c "\.md" docs/sessions/index.md

# Findings in nav
ls docs/findings/exp*.md | wc -l
grep -c "findings/" mkdocs.yml
```

### 6. Broken file references

Check that files mentioned in docs actually exist.

```bash
# Find markdown links and check targets
grep -roh '\[.*\](\.\./[^)]*\.md)' docs/ | grep -o '(.*\.md)' | tr -d '()' | sort -u | while read f; do
  resolved="docs/$f"
  [ ! -f "$resolved" ] && echo "BROKEN: $f"
done
```

### 7. Stale TODO items

Check if TODO.md has items marked done that aren't actually done, or open items that were completed.

```bash
# Open items
grep -c "^\- \[ \]" TODO.md

# Compare with closed issues
gh issue list --repo cybertronai/SutroYaro --state closed --json title --jq '.[].title' | head -10
```

## When to run

- After merging 2+ PRs
- Before preparing a weekly catch-up
- Before a meeting
- When someone says "is this up to date?"

## Output

List every stale item with: file, line, what it says, what it should say. Don't fix anything automatically. Report the findings so the user can decide what to update.
