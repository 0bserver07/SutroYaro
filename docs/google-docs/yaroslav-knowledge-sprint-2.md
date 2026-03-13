!!! info "Cross-references"
    **Source**: [Google Doc](https://docs.google.com/document/d/105FkE_U5_cXA1o8sxMGrj1NuTxysfSBTZsLB7vift-M/edit?tab=t.0) · [Meeting #8 summary](../meetings/notes.md#meeting-8-09-mar-26-demos-and-roadmap)
    **Related**: [Sprint 1](yaroslav-technical-sprint-1.md) · [The Bigger Picture](bigger-picture.md)

# Take-aways

Cache lemma means LRU cache is within 2x factor of optimal cache. Because optimal caching strategy is hard to compute, just assume LRU cache for everything. Continuous LRU cache and and data movement complexity of Ding are promising ([[paper](https://arxiv.org/abs/2312.14441)])

ARD is likely a sufficiently good metric. It's impossible to drive ARD to zero without improving energy efficiency of algorithms. Data Movement Cost is likely a better metric, by virtue of being known in the literature, and directly corresponding to energy cost for a specific kind of physical 2D memory layout.

# Notes 

Motivation: We have used average reuse distance before but I had some doubts about how interesting it is. I want to avoid metrics which, when optimizing, could create an algorithm with absolutely no relevance either to the physical world or to the literature world (meaning scientific literature). I went on a fishing expedition to remind myself how energy works.

Now we have a hierarchy of caches but there is a great simplification which is known as the square root rule. Basically the size of the cache and the cost, both in terms of latency and in terms of energy, can be approximated as the square root of the square root of the size of the cache. The reason for this can be tracked down to basic physics. If you imagine you have something of size k, the size of the wire to access that cache is, it might be, k squared, as long as you lay it out on a two-dimensional grid.

The second great simplification is that the last recently used cache is within a factor of 2 of the optimal cache. An optimal cache, imagine if you're using some piece of data and then you have a capacity conflict; you might evict this piece of data if it has been used before. An optimal cache can see the future and it would just say, \"Hey don't evict it cause we might use it.\" Because of the simplification we just can assume we use the last recently used cache.

With these two simplifications we can use a much simpler heuristic. The reason it is interesting is because:

1. Conceptually an algorithm that optimizes this could be built on chip. We just use wires to collect the rings and concentric circles and that's what will be the energy cost that we will have.

2. The second reason it is interesting is because it connects to existing literature of DING.

\-\-\-\--

Goal is to get all the relevant background to double  check on the right proxy metric. I used Average Reuse Distance, check if it's a good metric. [[notability](https://notability.com/n/2Ip3HX14t3Kfb~QbS6CCGT)]

Ultimate metric is energy efficiency on a common GPU like H100, However, iterating on an actual GPU is too slow, so we need to find a simpler proxy. Understand better the cache issue and iterate on various proxy metrics.

Research GPU availabilities and costs: [[https://chatgpt.com/share/69ac8b16-3e54-8011-8b4c-9b76a357da9e](https://chatgpt.com/share/69ac8b16-3e54-8011-8b4c-9b76a357da9e)]

- morning, Iterate on understanding of proxies in WebGPUmode ([[gemini](https://gemini.google.com/share/7b151cbf132e)])

- Summarize main concepts in \"energy proxy metrics\" [[notability](https://notability.com/n/2Ip3HX14t3Kfb~QbS6CCGT)]

        - kinetic locality Reflects the energy needed to keep memory resident between instructions. Probably negligible, so ignore that for now.

        - Ultimate metric appears to be the reuse distance histogram, but that's not a single number.

        - Average Reuse Distance remains important.

- look into working set size and ARD in WebGPUmode ([[gemini](https://gemini.google.com/share/7b151cbf132e)])

Videos: 

[[Herb Sutter @ NWCPP: Machine Architecture: Things Your Programming Language Neve](https://www.youtube.com/watch?time_continue=1&v=L7zSU9HI-6I&embeds_referring_euri=https%3A%2F%2Fgemini.google.com%2F&embeds_referring_origin=https%3A%2F%2Fgemini.google.com&source_ve_path=Mjg2NjY)] ([[slides](https://nwcpp.org/talks/2007/Machine_Architecture_-_NWCPP.pdf)])

[[code::dive conference 2014 - Scott Meyers: Cpu Caches and Why You Care](https://www.youtube.com/watch?v=WDIkqP4JbkE)] ([[slides](https://www.aristeia.com/TalkNotes/codedive-CPUCachesHandouts.pdf)])

[[14. Caching and Cache-Efficient Algorithms](https://www.youtube.com/watch?v=xDKnMXtZKq8)]

## Herb Sutter 

- Herb Sutter video: ground truth of memory wall

        slides: [[https://nwcpp.org/talks/2007/Machine_Architecture\_-\_NWCPP.pdf](https://nwcpp.org/talks/2007/Machine_Architecture_-_NWCPP.pdf)]

        video: [[Herb Sutter @ NWCPP: Machine Architecture: Things Your Programming Language Neve](https://www.youtube.com/watch?time_continue=1&v=L7zSU9HI-6I&embeds_referring_euri=https%3A%2F%2Fgemini.google.com%2F&embeds_referring_origin=https%3A%2F%2Fgemini.google.com&source_ve_path=Mjg2NjY)]

        [[notebook](https://notebooklm.google.com/notebook/9520a922-a76b-4a54-9c6d-83e48cc6d94b)] 

        (99% of software complexity goes into hiding latency, due to Memory Wall. Little Law)

- updating latency diagram to modern times [[gemini](https://gemini.google.com/app/36e244dc6f099e26)] (no change in latency)

- How GPUs changed over last 10 years: [[https://gemini.google.com/share/38e52f298618](https://gemini.google.com/share/38e52f298618)]

[[https://nwcpp.org/talks/2007/Machine\_](https://nwcpp.org/talks/2007/Machine_Architecture_-_NWCPP.pdf)]

[embedded image]

[[https://youtu.be/xDKnMXtZKq8?si=atySjQ79DdWimyGn](https://youtu.be/xDKnMXtZKq8?si=atySjQ79DdWimyGn)]

## Scott Meyers: CPU Caches 

[[code::dive conference 2014 - Scott Meyers: Cpu Caches and Why You Care](https://youtu.be/WDIkqP4JbkE?si=3BmmvyJrp2M4DyNV&t=1036)]

[embedded image]

## Danfu TogetherAI kernels course 

danfu tutorials - [[together-kernels](https://drive.google.com/drive/folders/1tnqtk6J9xNEUJ9AR6wwn4Gx1_IBxSMqC)](backup [[here](https://drive.google.com/open?id=1HTaS3r7obEuZQwq-OwyMBnXsIhKsXvGL&usp=drive_fs)])

12:34 \-- Plan now is to look at the MIT course and think about the metric again.

## 6.172, Performance Engineering of Software Systems 

[[notebook](https://notebooklm.google.com/notebook/6a4ada95-5490-4258-af5a-0799af2a480a)] [[course](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/pages/lecture-slides/)]

lecture 14

[[https://www.youtube.com/watch?v=xDKnMXtZKq8&t=329s](https://www.youtube.com/watch?v=xDKnMXtZKq8&t=329s)]

[[https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/329bfc6e1808c375afa517feb3c4c273_MIT6_172F18_lec14.pdf](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/329bfc6e1808c375afa517feb3c4c273_MIT6_172F18_lec14.pdf)]

LRU lemma means we can assume LRU model

[embedded image]

lecture 15

[[https://www.youtube.com/watch?v=xwE568oVQ1Y&t=96s](https://www.youtube.com/watch?v=xwE568oVQ1Y&t=96s)]

[[https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/cef17369f91d3140409f2be4ad9246a4_MIT6_172F18_lec15.pdf](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/cef17369f91d3140409f2be4ad9246a4_MIT6_172F18_lec15.pdf)]

Log

12:58 \-- How about checking the energy cost in Bill Daily [[notebook](https://notebooklm.google.com/notebook/50847971-3eef-4033-901c-616bb215577e)]

13:04 - Bill Daly energy research [[report](https://docs.google.com/document/d/183zsZkdCWTvwagHXYfIm7wlOarstwNdFdHu3MFPZQeE/edit?usp=sharing)]

13:04 \-- MIT course talks about cache lines, but modern GPUs use thread memory coalescing. (pnpp [[notebook](https://notebooklm.google.com/notebook/06c3f948-2886-4f65-aa56-576e316be132)])

TLDR; LRU replacement policy is within 2x of optimal

[embedded image]

## Stephen Jones: How GPU computing works  

[[Notebook](https://notebooklm.google.com/notebook/a2151504-f72c-44d3-be35-7016ca0ac022)]

[[https://www.youtube.com/watch?v=3l10o0DYJXg](https://www.youtube.com/watch?v=3l10o0DYJXg)]

calculates how many threads are needed to keep cores utilized, based on latency+bandwidth numbers

5x oversubscription for A100

## James Demmel, Communication-avoiding algorithms 

[[notebook](https://notebooklm.google.com/notebook/6e5ab97c-600a-42ea-a071-53e99ae34428)]

[[https://simons.berkeley.edu/sites/default/files/docs/827/demmelslides.pdf](https://simons.berkeley.edu/sites/default/files/docs/827/demmelslides.pdf)]

[[James Demmel: Communication-Avoiding Algorithms for Linear Algebra, Machine Learning and Beyond](https://www.youtube.com/watch?v=sY3bgirw--4)]

[[Communication Avoiding Algorithms for Linear Algebra and Beyond](https://www.youtube.com/watch?v=_FqYHIsuunw)]

[embedded image]

Research communication avoid algorithms for learning

[[https://chatgpt.com/c/69acbb1e-4e74-8322-a0fb-349e5961cf40](https://chatgpt.com/c/69acbb1e-4e74-8322-a0fb-349e5961cf40)]

## Ranking heuristics for LRU cache reuse 

16:22 Ask about other heuristics [[gemini](https://gemini.google.com/app/cf721afd6b73481a)] [[notebook](https://notebooklm.google.com/notebook/6e5ab97c-600a-42ea-a071-53e99ae34428)]

must rely on bytes touched (Stack Distance) rather than instructions elapsed (Reuse Distance)

- Cache Complexity (requires cache-size B)

[embedded image]

- Working Set Size (requires execution window T)

- Reuse Distance Profile (a histogram of reuse distances)

Logarithmic Area Under the Curve (Log-AUC)

The Harmonic Mean of Stack Distance (or Stack Distance)

Follow-ups to Demmel

[[https://chatgpt.com/c/69acc320-d2d8-8322-be45-2b5d6dc71997](https://chatgpt.com/c/69acc320-d2d8-8322-be45-2b5d6dc71997)], ([[shared](https://chatgpt.com/share/69addb3c-c0ec-8011-bdd1-346957d5402f)]) [[gemini](https://gemini.google.com/app/37afdf6fbce7c890)] ([[shared](https://gemini.google.com/share/0e03cd4abbdb)])

Average 

Reuse Distance (count instructions between accesses)

Stack Distance 

[embedded image]

Smooth hardware-agnostic heuristic \-- [[https://chatgpt.com/c/69accbbc-c260-8321-9078-a8d31d0da609](https://chatgpt.com/c/69accbbc-c260-8321-9078-a8d31d0da609)]

Energy roofline (arch line) \-- [[https://perso.ens-lyon.fr/christophe.alias/evalM2/choi2013-archline-ipdps.pdf](https://perso.ens-lyon.fr/christophe.alias/evalM2/choi2013-archline-ipdps.pdf)]

Follow-up paper - [[https://arxiv.org/abs/2509.20189](https://arxiv.org/abs/2509.20189)]

Analyze the paper and consider relevant heuristics \-- [[https://chatgpt.com/c/69acce0b-97ac-8324-b82e-31af148de335](https://chatgpt.com/c/69acce0b-97ac-8324-b82e-31af148de335)]

Sunday 9am, energy costs , 32 byte min but need 2KB request to amortize row activation energy bill daly [[notebook](https://notebooklm.google.com/notebook/50847971-3eef-4033-901c-616bb215577e)] 

## Ding: Data movement complexity 

DMC4ML: Data Movement Complexity for Machine Learning

[[https://notebooklm.google.com/notebook/78b74c92-0a68-43c8-8f75-e1a480d75983](https://notebooklm.google.com/notebook/78b74c92-0a68-43c8-8f75-e1a480d75983)]

implementing continuous LRU 

[[gemini](https://gemini.google.com/app/dd521e6c449eaa16)]

[[deepthink](https://gemini.google.com/app/46aec643d10abc8d)] \-- python implementation ([[shared](https://gemini.google.com/share/7aefdb4a99b2)])

[[chatgpt](https://chatgpt.com/c/69ada12a-bd68-8323-b846-82a31aad6746)]
