# Weekly Catch-Up: Mar 16-22, 2026

Prepared Sunday March 22, 3:09 PM. Tomorrow is Meeting #10.

## Sync Status

| Source | Result |
|--------|--------|
| Google Docs | 17 docs synced (added Meeting #9 notes) |
| Telegram | 861 messages across 6 topics |
| GitHub | 0 open PRs, 8 open issues |

---

## Meeting #9 Happened (Mar 16)

Yaroslav presented the roadmap. Participants: Yaroslav, Moorissa Tjokro (SPC, robotics/autonomous vehicles), Anastasiia Zhiboedova (ML Engineer, Amazon AGI), Michael Keating (data center cooling, attending NVIDIA GTC), Jonathan Belay, Yad Konrad (async via pre-recorded video), JackJack Ganbold (SPC), Andrew, Preston Schmittou.

Key outcomes:

- **Metric shift: ARD to DMC** (Data Movement Complexity, Ding et al. arXiv:2312.14441). The new homework is to optimize DMC instead of ARD. DMC uses sqrt(stack_distance) per access, which maps to physical wire-length energy on a 2D memory layout.
- **Meta-goal**: iterate on the *process* of going from "metric + problem specification" to a fast sequence of experiments. Not just solving the problem, but making the solving fast.
- Meeting video: [YouTube](https://www.youtube.com/watch?v=vdQ3NkEiOt8)
- AI slides: [Sutro_Efficiency_Synthesis.pdf](https://drive.google.com/file/d/1GS0zeBfNhyW7Qv76ItbFD1pKLfja7zWm/view)

---

## New Ideas from Telegram This Week

### Potential high-profile visitors (chat-yaroslav, Mar 21)

- **Lukas Kaiser** left OpenAI, doing open-source research. Plans to stop by the **Mar 30 meeting**.
- **Alec Radford** also doing OSS research now. Yaroslav wants to involve both.

### RL environment framing (chat-yaroslav, Mar 21)

- Yaroslav wants to wrap our algorithmic challenges into RL environments and give them to companies like Anthropic. If it makes Claude better, that accelerates our own auto-research loops.
- PrimeIntellect has a research grants program (compute + stipends) for novel environments.
- Yad noted that our 33 experiments are basically an answer key -- did the agent rediscover GF(2)? Did it figure out local learning rules fail? That's richer signal than most RL envs.

### Discrete ML / Wolfram (general, Mar 19-20)

- Yaroslav shared Wolfram's work on training neural nets with pure discrete Boolean logic (AND/XOR grids). No backprop, no floats.
- Seth Stafford: "neural networks are just a quantization of random forests. You recover the random forest in a semi-classical limit."
- 8-bit integer multiply is 5x less Joules than 16-bit FP. 8-bit integer adds: ~50x cheaper.

### Repo / logistics (chat-yad, Mar 21)

- Yaroslav asked if SutroYaro can be designated **Public Domain**. Yad said yes.
- Yaroslav visiting **Manhattan Mar 24-30** (this coming week). Wants to meet up.
- Video quality note: videos came through at 720p, Yaroslav asked for 4k next time.

### Michael's Autoresearch fork (challenge #1, Mar 16)

- Michael forked Karpathy's new Autoresearch, pointed it at sparse parity, asked Opus to use "unconventional or ancient mathematical theories" to avoid leaning on conventional wisdom.

---

## DMC Infrastructure Inventory

Before running experiments, we audited what exists for the DMC metric shift.

| Component | Status | Notes |
|-----------|--------|-------|
| `tracker.py` (DMC formula) | DONE | `sum(size * sqrt(distance))` per Ding et al. |
| `cache_tracker.py` (LRU + DMC) | DONE | Inherits DMC, adds cache simulation |
| `harness.py` (all 5 methods) | DONE | GF2, KM, SGD, SMT, Fourier all report DMC |
| `fast.py` (quick iteration) | MISSING | Zero tracker integration (#15) |
| 33 experiment files | MISSING | None instantiate MemTracker for DMC |
| `scoreboard.tsv` | PARTIAL | DMC column exists, only 5 of 35 rows filled (#16) |
| DMC visualization / plotting | MISSING | Nothing exists (#18) |
| Cross-method DMC comparison | MISSING | CLAUDE.md shows ARD table, no DMC equivalent |

---

## GitHub Issues (16 open, no new PRs)

### Homework (due tomorrow)

| Issue | Title |
|-------|-------|
| [#17](https://github.com/cybertronai/SutroYaro/issues/17) | DMC baseline sweep: measure all methods |
| [#22](https://github.com/cybertronai/SutroYaro/issues/22) | DMC optimization experiment: beat baseline on at least one method |

### Infrastructure (DMC shift)

| Issue | Title |
|-------|-------|
| [#15](https://github.com/cybertronai/SutroYaro/issues/15) | Add tracker integration to fast.py |
| [#16](https://github.com/cybertronai/SutroYaro/issues/16) | Backfill scoreboard.tsv with DMC values |
| [#18](https://github.com/cybertronai/SutroYaro/issues/18) | DMC visualization and plotting |
| [#6](https://github.com/cybertronai/SutroYaro/issues/6) | Compare DMC vs ARD vs real GPU joules |

### Strategic

| Issue | Title |
|-------|-------|
| [#19](https://github.com/cybertronai/SutroYaro/issues/19) | Prototype sparse parity as RL/eval environment |
| [#20](https://github.com/cybertronai/SutroYaro/issues/20) | Add Public Domain license |
| [#21](https://github.com/cybertronai/SutroYaro/issues/21) | Prep for Mar 30 meeting: Lukas Kaiser + Alec Radford visiting |

### Existing (from before)

| Issue | Title |
|-------|-------|
| [#4](https://github.com/cybertronai/SutroYaro/issues/4) | Push SGD under 10ms on sparse parity |
| [#5](https://github.com/cybertronai/SutroYaro/issues/5) | Test agent loop on sparse sum and sparse AND |
| [#7](https://github.com/cybertronai/SutroYaro/issues/7) | Add more task variations |
| [#8](https://github.com/cybertronai/SutroYaro/issues/8) | Agent complexity budget |
| [#9](https://github.com/cybertronai/SutroYaro/issues/9) | Modal integration for nanoGPT energy baseline |
| [#13](https://github.com/cybertronai/SutroYaro/issues/13) | Agent compatibility layer |
| [#14](https://github.com/cybertronai/SutroYaro/issues/14) | Agent notification bridge |

---

## What's Due Tomorrow (Meeting #10)

From the Meeting #9 homework:

1. **Get agents to improve sparse parity using DMC** (not ARD) as the energy proxy (#17, #22)
2. **Iterate on prompts and meta-approaches** -- make it fast to go from "metric spec + problem spec" to a sequence of experiments

Our DMC metric is already implemented (task #1 in docs/tasks/INDEX.md is DONE). But we haven't run experiments optimizing DMC yet. The CacheTracker/MemTracker already tracks DMC alongside ARD (baseline: ARD 4,104 / DMC 300,298).

---

## Task Lists

### Homework Tasks (Due Monday)

See [007-homework-meeting10.md](../tasks/007-homework-meeting10.md) for full breakdown.

- [ ] Run DMC baseline sweep across top methods (#17)
- [ ] Run at least one DMC optimization experiment (#22)
- [ ] Prepare results summary for presentation

### Infrastructure (This Week)

- [ ] Add tracker integration to fast.py (#15)
- [ ] Backfill scoreboard.tsv with DMC values (#16)
- [ ] Create DMC visualization scripts (#18)

### Strategic (Before Mar 30)

- [ ] Prototype sparse parity as RL env (#19)
- [ ] Add Public Domain license (#20)
- [ ] Prep for Mar 30 meeting (#21)
- [ ] Compare DMC vs ARD rankings (#6)
