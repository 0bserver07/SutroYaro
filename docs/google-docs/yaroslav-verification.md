\!\!\! info "Cross-references"
    **Source**: [Google Doc](https://docs.google.com/document/d/1jNN7NOssLRjrF6H0DGN61q5nGtuZVwnOwiTiLLToj7M/edit?tab=t.0) · [Meeting #8 summary](../meetings/notes.md#meeting-8-09-mar-26-demos-and-roadmap)
    **Related**: [GF(2) experiment](../findings/exp_gf2.md) · [Survey](../research/survey.md) · [Visualizer](https://gf-2-sparse-parity-solver-400699997518.us-west1.run.app/)


Following #7 homework, from [[Sutro Group](https://docs.google.com/document/d/1B9867EN6Bg4ZVQK9vI_ZqykZ5HEtMAHJ7zBGGas4szQ/edit?tab=t.0#heading=h.mw2co1cau175)]

TLDR; there's a 10,000x faster solution that is discovereable by agents.

Following #7 homework, from [[Sutro Group](https://docs.google.com/document/d/1B9867EN6Bg4ZVQK9vI_ZqykZ5HEtMAHJ7zBGGas4szQ/edit?tab=t.0#heading=h.mw2co1cau175)]

TLDR; there's a 10,000x faster solution that is discovereable by agents.

# Raw Log 

### Michael Keating 

 [[sutro_challenge_3_sparce parity results.docx](https://docs.google.com/document/d/13uAQfG_ola3vt1hHFo3A8ThUeV-nBVQK/edit?rtpof=true&sd=true&tab=t.0)]

Questions:

- What was the approach that was used? How hard was it?

- dug out evolutionary search, looks like it focused on methods from the 80s

### Germain Brillone 

Germaine Brillon:

\> Germain: my AI research team results so far :

You found a confirmed admissible improvement (multi-seed) under the same contract/instrumentation:

• Baseline ARD: \~48.1--48.9 across seeds

• Depth1 + Hidden64 ARD: \~33.1--34.5 across seeds

• Accuracy gate passed (≥0.90) and time was well under 2s.

But the winning change coincides with a big drop in total_accesses (≈92k → 64k). That might be legitimately "less work" (fewer layers/params/state)\...more research needed \> Germain: what we tried :

• AdamW → SGD (with LR=0.05): failed the accuracy gate, so it was rejected (no trace).

Interpretation: SGD might still be a win, but you need a LR/momentum that passes the 200-step budget gate.

• Hidden 32 → 16: gate passed, but ARD got worse than baseline.

Interpretation: smaller hidden didn't improve locality in your tracer's view; it may have changed reuse patterns unfavorably.

• Depth2 → Depth1, Hidden64: gate passed, and ARD dropped a lot (baseline \~48 → \~33--35 across seeds).

Interpretation: the winning change reduced the number of layers/objects touched per step, which reduces the number of distinct "intervening" objects between reuses --- so the reuse-distance proxy improves strongly. 

Questions:

- What was the approach? What were the pros and cons, agent failings?

### Yad Konrad 

[[https://github.com/0bserver07/SutroYaro](https://github.com/0bserver07/SutroYaro)]

\-\-\-\-\--

13:15: Remember the objective \-- The objective of this is to see if the agents are moving in the right direction and get some higher-level lessons from this.

One lesson from Germain's experiment is the importance of using the right metric, and for metric code to be isolated from agent modifications. This motivated [[Yaroslav Knowledge sprint #1](https://docs.google.com/document/d/105FkE_U5_cXA1o8sxMGrj1NuTxysfSBTZsLB7vift-M/edit?tab=t.0)]. 

13:45 Goal is to go over the results and make sure they are valid. The overall goal is to check if the direction is promising towards the goal of creating lower-energy GPT-2 level run.

Verify Yad's result

pushd \~/git0

git clone https://github.com/0bserver07/SutroYaro

cd SutroYaro

python3 src/sparse_parity/experiments/exp_gf2.py

works

Use AI to verify result. 1, use the original prompt ([[Benchmark for Energy-Efficient Learning via Sparse Parity Task](https://docs.google.com/document/d/1cs8DsWnz9CRoit6JknPavCfNlOV5DfjsSPVGyWH415s/edit?tab=t.0)]), and Yad's solution, ask to provide feedback

Open Yad's folder in antigravity, ask it to make \"python3 src/sparse_parity/experiments/exp_gf2.py\" self-contained. This will then be used for verification.

Refactor into standalone and push to github: [[https://github.com/cybertronai/sutro/blob/main/exp_gf2_standalone.py](https://github.com/cybertronai/sutro/blob/main/exp_gf2_standalone.py)]

Prompt:

Take the following specification [[sutro group challenge #1: sparse parity](https://docs.google.com/document/d/16eeltCaTpiiM1t_m_5BSxRnqxoMoiJ-xn4cy0x-TFgc/edit?tab=t.0)]and the following solution: [[https://github.com/cybertronai/sutro/blob/main/exp_gf2_standalone.py](https://github.com/cybertronai/sutro/blob/main/exp_gf2_standalone.py)] , and the following output after running this solution. Give a critical analysis of the solution, whether it satisfies the constraints of the problem. Give potential pitfalls such as overfitting to the task and whether this advances us towards the goal.

Comment on the higher level strategy (agentic loops in And whether the lessons of this approach (agentic loop) [[https://github.com/0bserver07/SutroYaro](https://github.com/0bserver07/SutroYaro)])

The overall goal: 

To advance us towards the goal of creating a more energy efficient way to solve a GPT-2-like task - Learn to predict the next Wikipedia character by training on wikitext-103, at lower energy on GPUs than transformer-based gradient descent methods, without sacrificing accuracy, generalization, or wall clock time. This is meant to be done with heavy assistance of agentic and AI methods, The bulk of the advances will consist of meta skills such as learning how to prompt and guide agents, how to set up evaluations and metrics.

Add source file, output file, homework spec, Sutro Group root document.

[embedded image]

DeepThink result: [[result](https://docs.google.com/document/d/1SxQ6XnnTQqZ_NxemVty5q7rhWP_Tz9cmUpZD0MlmZmU/edit?tab=t.0)]

Compare the energy efficiency of the standalone scripts to the baseline sparse parity benchmark. I'm interested in the average reuse distance. sparse_parity_benchmark.py vs exp_gf2_standalone.py

15:05 \-- compare energy efficiency using claude

\"[[https://github.com/0bserver07/SutroYaro](https://github.com/0bserver07/SutroYaro)]\"

prompt: Compare the energy efficiency of the standalone scripts to the baseline sparse parity benchmark. I'm interested in the average reuse distance. sparse_parity_benchmark.py vs exp_gf2_standalone.py

prompt: Modify the two versions to ensure memory tracking is tracked accurately, and compare memory efficiency of the two approaches.

prompt: Make a standalone Python file which runs both of these benchmarks and reports results side-by-side.

[[https://github.com/cybertronai/sutro/blob/main/compare_reuse_distance.py](https://github.com/cybertronai/sutro/blob/main/compare_reuse_distance.py)]

[[gemini](https://gemini.google.com/app/9622ae290dae0047)] \-- [[shared](https://gemini.google.com/share/31434aff46a2)]

[[deepthink](https://gemini.google.com/app/3519865b7cd7eaf4)] \-- [[shared](https://gemini.google.com/share/ef71b135badc)] \-- [[doc](https://docs.google.com/document/d/13aCtuFKF_IDC9jmEx68fI7xsKL_Fupfd1oNH6GIO5Mc/edit?tab=t.0)]

Asking for more in-depth analysis of energy usage

depethink [[https://gemini.google.com/app/3519865b7cd7eaf4](https://gemini.google.com/app/3519865b7cd7eaf4)] (failed, try in new [[deepthink](https://gemini.google.com/app/55db23c408db6402)])

gemini: [[https://gemini.google.com/app/9622ae290dae0047](https://gemini.google.com/app/9622ae290dae0047)]

notebook: [[https://notebooklm.google.com/notebook/a606c630-dbf2-44d6-852f-ed43bf411b82](https://notebooklm.google.com/notebook/a606c630-dbf2-44d6-852f-ed43bf411b82)]

understanding visualization : [[https://aistudio.google.com/apps/ec58fdc1-db78-4011-9abe-f79af0545f24?showPreview=true&showAssistant=true](https://aistudio.google.com/apps/ec58fdc1-db78-4011-9abe-f79af0545f24?showPreview=true&showAssistant=true)]

published [[app](https://gf-2-sparse-parity-solver-400699997518.us-west1.run.app/)]

Walk me through the solution \-- [[https://gemini.google.com/app/ac245afae642cac0](https://gemini.google.com/app/ac245afae642cac0)] [[shared](https://gemini.google.com/share/896ba5270dd6)]

DeepThink cache analysis from [[deepthink](https://gemini.google.com/share/cd3e712c8ae9)], [[doc](https://docs.google.com/document/d/1VRS5CqqldGhw9BR-Io9L9ncMoRHTl6EN9TwvjWkYnls/edit?tab=t.0)] (can do cache-oblivious recursive elimination)

# Final Comparison 

 [[https://github.com/cybertronai/sutro?tab=readme-ov-file#sgd-vs-gf2-gaussian-elimination-2026-03-09](https://github.com/cybertronai/sutro?tab=readme-ov-file#sgd-vs-gf2-gaussian-elimination-2026-03-09)]
