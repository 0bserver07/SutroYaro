# Sutro Group Challenge #1 

sparse parity

parent: [Sutro Group](https://docs.google.com/document/d/1B9867EN6Bg4ZVQK9vI_ZqykZ5HEtMAHJ7zBGGas4szQ/edit?tab=t.0)

Motivation:

Go back to 1960s and reinvent AI from scratch. We have an advantage over people in the 1960-1980s who were doing it because:

a) know where we are going

b) have AI agents

c) our phones are more powerful than 1980s university clusters

We can incorporate the following concrete lessons:

1. next-character prediction is the most important application

2. AI is bottlenecked by energy

3. memory cost is the biggest contributor to energy use (see bill daly [talk](https://youtu.be/rsxCZAE8QNA?si=8-kIJ1MuhxChRLgW&t=2457))

Emmet, Germaine, Andy, Seth were able to improve energy efficiency on some tasks. Microgpt task was interesting (by virtue of being popular), Emmett was able to drive energy usage 2x using his Aster agentic loop framework however iteration time was adding friction (3 minutes per run).

Now it would be useful to practice starting from the other direction. Instead of existing application (like nanogpt), take a simplest possible learning task and practice solving it with the energy in mind 

Goal: practice inventing energy efficient learning algorithms for a simplest non-trivial learning task. 

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

Estimate memory energy usage. To avoid thinking about particular cache sizes, focus on a specific metric which strongly correlates with memory energy use \-- Average Reuse Distance. When Average Reuse Distance is small, data can be kept in small energy-efficient cache. Otherwise it goes to expensive external memory. ([interactive tutorial](https://ai.studio/apps/eca3f37a-175a-4713-bb17-622b24e17d3a) on reuse distance)

Sample challenge 2: prompt the model to improve the algorithm to improve Average Reuse Distance

## Scaling up 

- increase difficulty to 3-bit parity task

- increase difficulty to 3-bit parity task with \"dirty bits\". IE, insert some bits which are irrelevant. This is known as the \"sparse parity task\". Scale up to 20 total bits with 3 relevant bits and 17 \"noise\" bits

[[embedded image removed]]

Here's how Yaroslav approached the first part (don't use it as gospel, there are many inefficiencies there) [Yaroslav Sutro technical sprint #1 02mar26](https://docs.google.com/document/d/1344Vld2n9-8B-OfeeqI5sP9fqPYCLQTglc9pSAmeeEM/edit?tab=t.0)

## Tips 

- make sure iteration time is small (ie, training + evaluation takes \<1 second at all times)

- change one thing at a time, either focus on correctness, or wall-clock time performance, or energy usage, keep the other factors fixed.

- priorize 1 correctness (solving the task to given accuracy), followed by wall-clock time (faster iteration) followed by energy usage

- take small steps and checkpoint your work (every saved solution should be correct + fast to execute)

Related materials: Bill Daly about energy use in GPUs

[https://youtu.be/rsxCZAE8QNA?si=8-kIJ1MuhxChRLgW&t=2457](https://youtu.be/rsxCZAE8QNA?si=8-kIJ1MuhxChRLgW&t=2457)

# The question to answer 

Can modern AI

1) make a learning algorithm to solve this simple learning task

2) improve (memory) energy usage?

3) what are the prompting strategies/approaches that are useful here?
