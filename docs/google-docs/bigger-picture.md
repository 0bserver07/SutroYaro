!!! info "Cross-references"
    **Source**: [Google Doc](https://docs.google.com/document/d/1oSTIM0h1-THnb3fZ6Cz5o86DQx6n44si1q4BkP2lfSA/edit?tab=t.0) · [Meeting #8 summary](../meetings/notes.md#meeting-8-09-mar-26-demos-and-roadmap)
    **Key concept**: 3-axis cube (process, metric, problem). Final exam: energy-efficient nanoGPT training.
    **Related**: [Challenge #1](challenge-1-sparse-parity.md) · [Yaroslav Knowledge Sprint 2](yaroslav-knowledge-sprint-2.md)

Yaroslav Sutro planning sprint #1

Objective

Solve the energy problem of AI. Similar mission to [[https://ml.energy/](https://ml.energy/)] .

Method

Blank slate rewrite. Draw on [[experience](https://docs.google.com/document/d/1PhArG_B1qrurqeXtqG41n_OorHw5J9kJcaZkhJOh7NU/edit?tab=t.0)] to prevent mistakes.

Thoughts

([[planning-sprint1.txt](https://drive.google.com/open?id=1Q2y9pdU4cT4Wn0MBE63XoSeg1xDbgwR-&usp=drive_fs)], [[planning-sprint2.txt](https://drive.google.com/open?id=1NmYwxYazoV5pXXEpV9D6I8J5iP9HB1fb&usp=drive_fs)], [[plannint-spring3.txt](https://drive.google.com/file/d/1xYjR7S4iSG0UeVRZzr-AS72FdsjQ4L-B/view?usp=sharing)], [[gemini](https://gemini.google.com/share/bef3c777a11b)])

Yaroslav Sutro planning sprint #1

Objective

Solve the energy problem of AI. Similar mission to [[https://ml.energy/](https://ml.energy/)] .

Method

Blank slate rewrite. Draw on [[experience](https://docs.google.com/document/d/1PhArG_B1qrurqeXtqG41n_OorHw5J9kJcaZkhJOh7NU/edit?tab=t.0)] to prevent mistakes.

Raw Thoughts

([[planning-sprint1.txt](https://drive.google.com/open?id=1Q2y9pdU4cT4Wn0MBE63XoSeg1xDbgwR-&usp=drive_fs)], [[planning-sprint2.txt](https://drive.google.com/open?id=1NmYwxYazoV5pXXEpV9D6I8J5iP9HB1fb&usp=drive_fs)], [[plannint-spring3.txt](https://drive.google.com/file/d/1xYjR7S4iSG0UeVRZzr-AS72FdsjQ4L-B/view?usp=sharing)], [[gemini](https://gemini.google.com/share/bef3c777a11b)])

Yaroslav Sutro planning sprint #1

Objective

Solve the energy problem of AI. Similar mission to [[https://ml.energy/](https://ml.energy/)] .

Method

Blank slate rewrite. Draw on [[experience](https://docs.google.com/document/d/1PhArG_B1qrurqeXtqG41n_OorHw5J9kJcaZkhJOh7NU/edit?tab=t.0)] to prevent mistakes.

Raw Thoughts

([[planning-sprint1.txt](https://drive.google.com/open?id=1Q2y9pdU4cT4Wn0MBE63XoSeg1xDbgwR-&usp=drive_fs)], [[planning-sprint2.txt](https://drive.google.com/open?id=1NmYwxYazoV5pXXEpV9D6I8J5iP9HB1fb&usp=drive_fs)], [[plannint-spring3.txt](https://drive.google.com/file/d/1xYjR7S4iSG0UeVRZzr-AS72FdsjQ4L-B/view?usp=sharing)], [[gemini](https://gemini.google.com/share/bef3c777a11b)])

# Mistakes 

1. Too simple of an objective

2. Suboptimization

3. Risk mismanagement

4. Building for the wrong future

[embedded image]

## Wrong objective 

Real life objectives are hard to measure, so we substitute simpler versions. If it's \"too simple\" we get Surrogate Endpoint Fallacy. Example: Google Radiology failure (mit [[review](https://www.technologyreview.com/2020/04/27/1000658/google-medical-ai-accurate-lab-real-life-clinic-covid-diabetes-retina-disease/#:~:text=In%20the%20system%20Thailand%20had,blocked%20or%20leaking%20blood%20vessels.)], Andrew Ng's [[thebatch](https://info.deeplearning.ai/the-batch-pandemic-triage-asias-ai-advantage-detecting-deepfakes-cleaning-oceans-music-industry-in-a-box)]). Other examples ([[chatgpt](https://chatgpt.com/share/69b0810f-f414-8011-8cb7-ed4c5d8bd5f5)], [[deepthink](https://gemini.google.com/share/bec8fe81a0b3)]).

Unlike radiology, we don't need clinical trials, so it's easier to test the real objective periodically. Energy dissipates as heat (computer has 100% heating efficiency [[experiment](https://www.pugetsystems.com/labs/articles/gaming-pc-vs-space-heater-efficiency-511/#:~:text=However%2C%20if%20your%20house%20is,heater%2C%20and%20much%20more%20entertaining.)]), so we can estimate energy usage by checking if the device is warm to the touch. Phones and laptops have GPUs, sanity check could be to compare two algorithms on a consumer device and checking the temperature manually.

## Risk management 

Going too far without positive feedback risks losing trust or [[sanity](https://en.wikipedia.org/wiki/Grigori_Perelman)].

Vincent post ([[medium](https://vanhoucke.medium.com/so-you-want-to-be-a-research-scientist-363c075d3d4c)], [[archive](https://archive.ph/CCKiq)]), Terry Tao mathematical [[disease](https://rjlipton.com/2009/11/04/on-mathematical-diseases/)].

## Suboptimization 

Once you partition the problem space, optimizing parts can either get stuck or make the overall objective worse.

Examples:

Train / test. (mixture of experts made training more efficient, inference less efficient)

Math / kernels design. 

Hardware lottery (sarah [[hooker](https://hardwarelottery.github.io/)]), machine learning systems are [[stuck in a rut](https://dl.acm.org/doi/10.1145/3317550.3321441)] (Paul Barham, Michael Isard)

To avoid \-- remove as many boundaries as possible. At some point, the problem becomes too big to make progress on, we are forced to partititon to make progress. If we are motivated by keeping the complexity manageable,  we can periodically re-adjust the partitioning. In contrast to the current paradigm which keeps problem partitioning fixed:

- math people vs kernels people

- training improvement vs inference improvements

- optimizer design vs architecture design

## Building for the wrong future 

Building software for the wrong hardware can take a while to fix. Python still uses GIL, 20 years after multiple cores were introduced. B-tree/LST tree transition due to HDD-\>SSH transition was 10 years delayed ([[notebook](https://notebooklm.google.com/notebook/61ba75d6-72cb-4229-bc87-5af19f2ed778)]).

The field of deep learning settled on the core algorithms in the 1980s when we had 1 thread and computation was the primary bottleneck. Today, the bottleneck is memory.

Designing for algorithms that optimize memory-energy will work for today's hardware, but could be obsolete if a different paradigm takes over.

- The memory wall trend makes this unlikely.

- Can be mitigated by focusing on the process to obtain the algorithm rather than algorithm itself

# Checking for \"signs of life\" 

Demonstrate energy-efficient training of Karpathy's [[nanoGPT](https://github.com/karpathy/nanoGPT)]. To be convincing, the result must improve in energy usage without sacrificing other important metrics, such as wall-clock time and accuracy.

This can be viewed as a final exam, it must be passed in order to proceed to \"the next class\". 

# Method 

Get from A to B in the following diagram

[embedded image]

This diagram is a cube with three orthogonal directions

1. Orange: Improve the process (optimizing given metric for a given problem)

2. Green: Make the metric more realistic.

3. Blue: Make the problem more realistic.

To keep complexity manageable, take small steps on axis at a time.

Examples

1. The Process

Yaroslav: [[Yaroslav Sutro technical sprint #1 02mar26](https://docs.google.com/document/d/1344Vld2n9-8B-OfeeqI5sP9fqPYCLQTglc9pSAmeeEM/edit?tab=t.0)], [[Yaroslav Verification Sprint #1](https://docs.google.com/document/d/1jNN7NOssLRjrF6H0DGN61q5nGtuZVwnOwiTiLLToj7M/edit?tab=t.0)]

Yad: [[Setup Sutro Yaro with Claude Code \| Sparse Parity](https://www.youtube.com/watch?v=h8dAU8yngxM)] [[survey](https://0bserver07.github.io/SutroYaro/research/survey/)] [[github](https://github.com/0bserver07/SutroYaro)]

Michael: [[sutro_challenge_3_sparce parity results.docx](https://docs.google.com/document/d/13uAQfG_ola3vt1hHFo3A8ThUeV-nBVQK/edit)]

2. The Metric [[Yaroslav Knowledge sprint #1](https://docs.google.com/document/d/105FkE_U5_cXA1o8sxMGrj1NuTxysfSBTZsLB7vift-M/edit?tab=t.0#heading=h.p686ct9trrt)]

3. The Problem [[sutro group challenge #1: sparse parity](https://docs.google.com/document/d/16eeltCaTpiiM1t_m_5BSxRnqxoMoiJ-xn4cy0x-TFgc/edit?tab=t.0)]
