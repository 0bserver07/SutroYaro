!!! info "Cross-references"
    **Source**: [Google Doc](https://docs.google.com/document/d/1344Vld2n9-8B-OfeeqI5sP9fqPYCLQTglc9pSAmeeEM/edit?tab=t.0) · Presented at [Meeting #7](meeting-7-notes.md).
    **Related**: [Challenge #1](challenge-1-sparse-parity.md) · [Sprint 1 findings](../findings/sprint-1-findings.md) · [Per-layer ARD](../findings/exp_c_perlayer_20bit.md) · [Cache-aware ARD](../findings/exp_cache_ard.md)
    **Code**: [cybertronai/sutro](https://github.com/cybertronai/sutro) · [sparse_parity_benchmark.py](https://github.com/cybertronai/sutro/blob/e132532f67f97f927d4700afb913e76d5cbdab02/sparse_parity_benchmark.py)

# Yaroslav Technical Sprint 1

02 Mar 2026

Sutro project

TLDR; 2.5 hours of work, simple workflow demonstrates automatic identification of a gradient fusion strategy that improves energy efficiency slightly. (cache reuse improved by 16%). A significant improvement requires fundamentally different learning algorithm (according to Claude). Candidate algorithms are the topic of Technical Sprint 2.

Github: [[https://github.com/cybertronai/sutro/](https://github.com/cybertronai/sutro/)]

2:58 start

Goal - Evaluate as a priority task, as drosophilla of \"energy-efficient training\". Try the process of using AI to reinvent learning algorithms, compare the ease at which it is possible to improve energy efficiency compared to the \"default method\"

2;59 \-- Get background of prosperity task from the [[notebook](https://notebooklm.google.com/notebook/3eb7b53e-168f-409c-a679-2fb009119e2e)]

Generate summary slides and look at the \-- [[Sparse_Parity_Optimization.pdf](https://drive.google.com/file/d/1nC5KbckpwLjeGynuS4wPntBoClfDHyqY/view)]

3:06 Finished reading the slides; make a plan. While making the plan, re-examined the goal.

Plan A:

- prototype a tiny network which learns parity using three bits.

- Extend this by adding more dirty bits, this is all using Standard methods with gradient descent.

- Come up with a measure of efficiency like Joules.

- Prompt an algorithm to improve this measure.

- Generate conclusions, re-examine the goal, and make a new plan.

3:09 Choosing to use Antigravity because agents are integrated

3:10 Think about the purpose of the meeting again. Decide it's \"put face to name, share results, get motivated to continue\"

3:11 Created \~/git0/sutro folder for files, saved Antigravity workspace there

3:11 Create a meta-prompt for claude using sparse parity [[notebook](https://notebooklm.google.com/notebook/3eb7b53e-168f-409c-a679-2fb009119e2e)]

3:14 created meta-prompt

I am interested in creating a simple toy example which learns prosperity tasks using standard gradient descent and neural networks. Create a prompt I can use with Google Claude 4.6 in order to build a self-contained example. I will build on this to create new methods and evaluate energy usage. Here's the ultimate goal of the effort \"Goal - Evaluate as a priority task, as drosophilla of \"energy-efficient training\". Try the process of using AI to reinvent learning algorithms, compare the ease at which it is possible to improve energy efficiency compared to the \"default method\"

\"

3:14 save prompt [[here](https://docs.google.com/document/d/1cs8DsWnz9CRoit6JknPavCfNlOV5DfjsSPVGyWH415s/edit?tab=t.0)]

3:15 Read the prompt and edit it.

reduce the difficulty n to 3, sample sizes to 10 from 1000

Change to full-page trainin, For simplicity of implementation

I've noticed the prompt implements energy wrong, but for now let it go, This can be optimized later.

3:19 This is taking more than a minute, so do a parallel implementation using Gemini.

3:23 Finding the default implementation takes more than one second to run, iterating to make it faster.

3:27 saving fast version ([[saved](https://colab.research.google.com/drive/1auWQjRgtrqyzef98wqq796sl927tCGMq)]), By using Save A Copy in Drive functionality

Initial observation: the loss decreases between the first epoch and the second epoc, which is unexpected. Training accuracy and testing accuracy both improve, which is good.

Now focus on writing sanity checks to make sure it's actually solving the correct things, Ask the model to change to do one example at a time.

Prompt: Modify this to do one example at a time, preceding in a cyclical fashion from epoch to epoch without randomizing the order, and generate plots of both training and test losses. 

3:32 Feeling annoyed that Claude Code 3.6 takes so long, switching to Sonnet 4.6

3:55 distracted by messenger errands. Now back.

[embedded image]

3:59  reduce the total runtime to be under 1 second. Ask Opus to optimize

Without changing the functionality of the program, identify ways to make it faster without sacrificing readability or extensibility. Ideally I wanted to run under one second. 

16:00 Do some estimation whether one second is a reasonable threshold. ([[gemini](https://gemini.google.com/share/45a7920017e2)]) The reference number is that Yandy Kun in the 80s was using a Spark 7 machine, and his experiments took one week to run.

3 day run should be expected to run in 90 according to [[gemini](https://gemini.google.com/share/45a7920017e2)], However, surely I'll look one iterated on much shorter rounds in the meanwhile, so keep \<1 second as the requirement for any method, this corresponds to about 1 hour runtime on spark 7.

16:09 upload and check into [[repo](https://github.com/cybertronai/sutro)], take a walk to think about next steps.

16:17 slack distractions. Now really going for a walk.

16:41 Back from the talk. During the talk I thought about ways of measuring energy on this toy task. Let us remember the simple measure which correlates to energy. Average Reuse Distance (Brainstorming [[session](https://gemini.google.com/share/c99ec90874da)]). ARD\<cache size Implies that most access is via cache and therefore cheap. Instead of determining certain cache sizes to match the realistic load, we can just report average reuse distance.

Plan B (Modifying the original plan A)

a. Refactor the existing learning algorithm to report average reuse distance

b. Calculate average reuse distance and do sanity checks

c. Improve or the average reduce distances from the top down manner.

d. Improving the average reduced distance from the bottom map manner

e. Sanity checks and compare.

16:47 First, simplify the network to make it amenable to rewriting

16:55 

[embedded image]

16:56 Try to make it even simpler by removing the biases; however, this breaks learning, so never mind.

17:04 Instructed Claude to make a pure Python version and checked it in, [[https://github.com/cybertronai/sutro/blob/e132532f67f97f927d4700afb913e76d5cbdab02/sparse_parity_benchmark.py](https://github.com/cybertronai/sutro/blob/e132532f67f97f927d4700afb913e76d5cbdab02/sparse_parity_benchmark.py)]

[embedded image]

17:04 Brainstorm with Gemini about average reuse distance. Because Claude code is so slow, I want to iterate a bit on the prompt first. \-- [[session](https://gemini.google.com/share/90bda930129a) ]

17:13 prompt claude to start adding reuse distance metrics

Decrease the number of steps to one example and compute metrics to help me track reuse distance. I am interested in tracking average reuse distance. In standard back propagation the first activation takes a while to be reused whereas the last activations have a short distance between creating it and accessing it. Change the code that allows me to track for each read axis, there is a reduce distance metric. The underlying motivation is to have a metric which is proportional to energy efficiency. For code with small reuse distance I can just keep this in LRU cache. For each memory read during this step I want to display the size of the buffer and also the \"distance\" from its generation. 

17:14

[embedded image]

17:20 this is based on the following memory reuse report

[embedded image]

17:21 Re-examining backprop access patterns ([[write-up](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)]) and suspicious about large values. Confirming that it counts individual reviews regardless of variable size, 

17:25  ask Opus to clean up and upload to [[github](https://github.com/cybertronai/sutro/tree/3059aad4abb49674fe85a2b9fcec50db62f48e9a)]

[embedded image]

Ask it to improve reuse distance \-- prompt with \"implement a way to improve reuse distance without affecting accuracy\"

The result is auomaticallyidentifyinga gradient fusion strategy 

[embedded image]

New sutro [[github](https://github.com/cybertronai/sutro/tree/d4fe8d95a0374c47d320dadb6916067ef31b7ddf)]

[large table/output removed]
  [embedded image]   [embedded image]
[large table/output removed]
[large table/output removed]

Asking Claude why the improvement is relatively modest. Obtained this answer:

W1 alone accounts for 6,000 of the 19,013 total floats read (32%), and its distance (\~15,000) didn't change at all --- it's read in forward and again at the very end of backward regardless of how we order the updates. Similarly b1 (2,000 floats, 11%) is unchanged.

The buffers we did improve (dW2: 16,005 → 3,002, db2: 18,005 → 5,002) contribute only 1,001 floats out of 19,013 total --- just 5% of the weighted sum. Even a 5× improvement on 5% of the total barely moves the needle.

In short: the fused update fixed the easy wins (gradient buffers), but the real bottleneck is the parameter tensors themselves (W1, b1), which are inherently read twice with the entire forward+backward pass in between. To fix those, you'd need a fundamentally different algorithm --- like computing Layer 1's backward and update before proceeding to Layer 2's forward (which would change the math), or doing per-layer forward-backward without propagating through the full network.
