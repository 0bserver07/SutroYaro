!!! info "Cross-references"
    **Source**: [Google Doc](https://docs.google.com/document/d/16eeltCaTpiiM1t_m_5BSxRnqxoMoiJ-xn4cy0x-TFgc/edit?tab=t.0) · Homework assigned at [Meeting #7](meeting-7-notes.md).
    **Related research**: [Sprint 1 findings](../findings/sprint-1-findings.md) · [All experiment results](../research/index.md) · [Yaroslav Sprint 1](yaroslav-technical-sprint-1.md) · [Literature review](../research/sparse-parity-literature.md)
    **Yad's reproduction**: [SutroYaro repo](https://github.com/0bserver07/SutroYaro) — 16 experiments, solved 20-bit k=3 in 0.12s

# Sutro Group Challenge #1

sparse parity

parent: [[Sutro Group](https://docs.google.com/document/d/1B9867EN6Bg4ZVQK9vI_ZqykZ5HEtMAHJ7zBGGas4szQ/edit?tab=t.0)]

Motivation:

Go back to 1960s and reinvent AI from scratch. We have an advantage over people in the 1960-1980s who were doing it because:

a) know where we are going

b) have AI agents

c) our phones are more powerful than 1980s university clusters

We can incorporate the following concrete lessons:

1. next-character prediction is the most important application

2. AI is bottlenecked by energy

3. memory cost is the biggest contributor to energy use (see bill daly [[talk](https://youtu.be/rsxCZAE8QNA?si=8-kIJ1MuhxChRLgW&t=2457)])

Emmet, Germain, Andy, Seth were able to improve energy efficiency on some tasks. Microgpt task was interesting (by virtue of being popular), Emmett was able to drive energy usage 2x using his Aster agentic loop framework however iteration time was adding friction (3 minutes per run).

Now it would be useful to practice starting from the other direction. Instead of existing application (like nanogpt), take a simplest possible learning task and practice solving it with the energy in mind 

Goal: practice inventing energy efficient learning algorithms for a simplest non-trivial learning task (also see  \"The question to answer\" below)

1. Use a neural network to solve task

2. Estimate energy use with memory use in mind

3. Change the algorithm to improve energy usage

4. Share AI tips with other people on ob

Here's an example of the XOR rule. (the example that Minsky used to trigger the AI winter)

11 -\> 0

10 -\> 1

01 -\> 1

00 -\> 0

10 -\> ?

Learning algorithm needs to fill in ?'s

Learning this is bit trivial, just need to memorization. Make it harder to use random negative numbers in place of 0 and arbitrary positive in place of 1. Here's is an example dataset with random positive/negative numbers matching above

0.3,  0.6 -\> 0

0.3, -0.5 -\> 1

-0.6, 0.1 -\> 1

-0.6, -0.2 -\> 0

-0.2, -0.2 -\> ?

Set of examples with ? output is known as the \"testing set\" and the challenge is to fill in the missing ? in all entries of the testing set. The remaining examples are known as \"training set\".

Sample challenge 1: generate larger training set, larger testing set, practice making a neural net algorithm which learns this rule from data. Aim for accuracy much larger than random, ie 90%.

Estimate memory energy usage. To avoid thinking about particular cache sizes, focus on a specific metric which strongly correlates with memory energy use \-- Average Reuse Distance. When Average Reuse Distance is small, data can be kept in small energy-efficient cache. Otherwise it goes to expensive external memory. ([[interactive tutorial](https://ai.studio/apps/eca3f37a-175a-4713-bb17-622b24e17d3a)] on reuse distance)

Sample challenge 2: prompt the model to improve the algorithm to improve Average Reuse Distance

## Scaling up 

- increase difficulty to 3-bit parity task

- increase difficulty to 3-bit parity task with \"dirty bits\". IE, insert some bits which are irrelevant. This is known as the \"sparse parity task\". Scale up to 20 total bits with 3 relevant bits and 17 \"noise\" bits

[embedded image]

Here's how Yaroslav approached the first part (don't use it as gospel, there are many inefficiencies there) [[Yaroslav Sutro technical sprint #1 02mar26](https://docs.google.com/document/d/1344Vld2n9-8B-OfeeqI5sP9fqPYCLQTglc9pSAmeeEM/edit?tab=t.0)]

## Tips 

- make sure iteration time is small (ie, training + evaluation takes \<2 seconds at all times)

- change one thing at a time, either focus on correctness, or wall-clock time performance, or energy usage, keep the other factors fixed.

- priorize 1 correctness (solving the task to given accuracy), followed by wall-clock time (faster iteration) followed by energy usage

- take small steps and checkpoint your work (every saved solution should be correct + fast to execute)

Related materials: Bill Daly about energy use in GPUs

[[https://youtu.be/rsxCZAE8QNA?si=8-kIJ1MuhxChRLgW&t=2457](https://youtu.be/rsxCZAE8QNA?si=8-kIJ1MuhxChRLgW&t=2457)]

# The question to answer 

Can modern AI

1) make a learning algorithm to solve this simple learning task

2) improve (memory) energy usage?

3) what are the prompting strategies/approaches that are useful here?

\<end-of-original-document\>

------------------------------------------------------------------------

===============================================================================

Yad's repro using Claude Code [[https://github.com/0bserver07/SutroYaro](https://github.com/0bserver07/SutroYaro)]

Germain's summary: 

## Energy-Efficient Learning Experiments (Clarified Brief) 

### Purpose 

Practice inventing energy-efficient learning algorithms on the simplest non-trivial learning tasks, using a memory-energy proxy (reuse distance) and an AI-assisted iteration loop. The point is less "XOR is hard" and more "build a repeatable workflow where AI suggests changes, you measure, and you iterate."

------------------------------------------------------------------------

## Success Criteria 

### Objective (what we optimize) 

Minimize Average Reuse Distance (ARD) (prefer true reuse distance; approximate if needed).

### Constraints (must remain satisfied) 

1.  Accuracy: test accuracy ≥ 90%\
    
2.  Iteration speed: end-to-end train + eval ≤ 2 seconds (or a similarly strict bound you set and keep fixed per phase)\
    

Formally:

\\min \\; \\text \\quad \\text \\quad \\text \\ge 0.90,\\; \\text \\le 2s

------------------------------------------------------------------------

## Phase 0 --- Setup (one-time) 

1.  Pick stack (e.g., PyTorch CPU first; later GPU).\
    
2.  Implement a repeatable runner that, for a given config, prints one line of metrics:\
    

- accuracy\
  
- runtime\
  
- ARD (or proxy)\
  
- any secondary stats you want (peak memory, allocations, etc.)\
  

3.  Define the measurement window for ARD (must be consistent):\
    

- Option A: training loop only\
  
- Option B: train + eval\
  \
  Pick one and keep it fixed.\
  

------------------------------------------------------------------------

## Phase 1 --- Baseline Task (2-bit parity / XOR) 

### Data definition (sign-encoded bits) 

We encode bits as real numbers:

- bit 0 → random negative value\
  
- bit 1 → random positive value\
  

For XOR (2-bit parity), label is:

- output = 1 if signs differ\
  
- output = 0 if signs match\
  

### Dataset generator 

Create a function that generates:

- training set: N samples\
  
- test set: M samples\
  \
  by sampling random sign-coded inputs and computing labels from the parity rule.\
  

### Model + training baseline 

Use a small neural net classifier (e.g., MLP). Keep it intentionally simple, but correct.

### Baseline target 

- Achieve ≥90% test accuracy\
  
- Keep runtime under the iteration bound\
  
- Record baseline ARD (or proxy)\
  

Deliverable: baseline code + a single "baseline metrics" record.

------------------------------------------------------------------------

## Phase 2 --- AI-Assisted Optimization Loop (core of the project) 

### Loop structure (repeat many times) 

1.  Measure current version\
    

- accuracy, runtime, ARD (or proxy)\
  

2.  Ask an AI agent to propose improvements\
    

- Provide: brief, constraints, current code excerpt (or summary), and last results\
  
- Ask for: 3--10 ranked modifications + rationale + expected impact\
  

3.  Select one change\
    

- "One change at a time" to preserve attribution\
  

4.  Implement and re-measure\
    
5.  Log results\
    

- change description\
  
- AI prompt used (or prompt template ID)\
  
- metrics before/after\
  

6.  Feed results back\
    

- Ask AI: "update hypothesis; propose next change"\
  

### What changes are in scope 

All changes are allowed, including:

- algorithmic: optimizer, update rule, truncations, checkpointing, etc.\
  
- systems: batch layout, fusion, in-place ops, caching, precision, avoiding allocations, streaming data\
  
- model: architecture variations, parameter sharing, alternative representations\
  
- measurement: using true reuse distance vs. a well-justified proxy (must remain consistent within a phase)\
  

Guardrail: The task definition, dataset generator, train budget, and eval protocol remain fixed within a phase, so improvements are comparable.

Deliverable: an experiment log showing multiple iterations and measured improvements.

------------------------------------------------------------------------

## Phase 3 --- Scaling Tasks (same loop, harder problems) 

### 3-bit parity 

Inputs: 3 sign-encoded floats

Label: 1 if an odd number of bits are 1, else 0

Goal: repeat baseline + optimization loop.

### Sparse parity (feature selection under noise) 

Inputs: total dimension D (e.g., 20)

Only k bits (e.g., 3) are relevant parity bits; remaining D--k are noise bits.

Generate samples where:

- relevant bits determine label via parity\
  
- noise bits are random and irrelevant\
  

Scale D upward (e.g., 20 → 50 → 100) while keeping k small.

Deliverable: results tables by task level showing (accuracy, runtime, ARD) and the best achieved configuration.

------------------------------------------------------------------------

## Phase 4 --- Transfer Mindset to "Real" Scale (future-facing) 

Keep asking: "Would this strategy help when training on GPUs with \~100M tokens?"

As the workflow matures, add GPU-relevant metrics (optional at first):

- bytes moved (HBM traffic proxies)\
  
- cache hit rates where available\
  
- kernel launch count\
  
- activation memory footprint\
  

------------------------------------------------------------------------

## Logging & Reporting (required) 

Maintain a simple run log (CSV/JSON/Markdown) with:

- date/time, git commit (or version id)\
  
- task phase (XOR / 3-bit / sparse)\
  
- config (model, batch, steps, precision, etc.)\
  
- AI prompt template used + the AI's recommended options\
  
- the chosen change\
  
- metrics before/after\
  

------------------------------------------------------------------------

## Notes / Best Practices 

- Keep iteration time small (\<2s) by design (small models, small datasets, fixed steps).\
  
- Prioritize in this order:\
  

1.  correctness (≥90% accuracy)\
    
2.  iteration speed\
    
3.  ARD / energy proxy improvements\
    

- Save checkpoints only when runs remain correct + fast.\
  
- Prefer true reuse distance; if not feasible, use a proxy consistently and document it.\
