!!! info "Cross-references"
    **Source**: [Google Doc](https://docs.google.com/document/d/1OXd_-RweVbHzjqTA2UF05mEs8iJZQiqKeU3QHU_DMdc/edit?tab=t.0) · [Meeting #6 summary](../meetings/notes.md#meeting-6-23-feb-26-presentations)
    **Key outcomes**: "Drosophila of learning" concept — need sub-1-second iteration task. Led to [Challenge #1: Sparse Parity](challenge-1-sparse-parity.md).
    **Related**: [Emmett's results](meeting-6-emmett-results.md) · [Pebbling games](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)

# Overview

[embedded image]

Emmett's results: [[sutro group \-- emmett's results 23feb26](https://docs.google.com/document/d/1DAwx_gohi6tomMPkb_fETAIuxIyHgLtC5OPD_qpGpqg/edit?tab=t.0)]

Germaine results: [[(sutro #6 germain) germaine.mp4](https://drive.google.com/file/d/1BzZLWCveiXRAXj2nAsYYDoPnPe8lDDki/view?usp=sharing)]

Yaroslav shown docs

- Horowitz Energy Problem [[https://notability.com/n/1OKuoEBjG78uifm8QxzqGm](https://notability.com/n/1OKuoEBjG78uifm8QxzqGm)]

- Bill Daly keynote: [[https://notability.com/n/1M_Wufsg\_\~T8TRNgYHe7L3](https://notability.com/n/1M_Wufsg_~T8TRNgYHe7L3)]

- Pebbling Games for maximum memory - [[https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)]

- Pebble Games [[notebook](https://notebooklm.google.com/notebook/e698f8b5-91c5-42aa-86aa-d6b88698fddf)]

        - Mojo GPU puzzles from csirak: [[link](https://github.com/modular/mojo-gpu-puzzles?tab=readme-ov-file)]

# Next steps 

- Find a way to iterate faster (maybe 500 parameter transformer that learns addition from [[tweet](https://x.com/DimitrisPapail/status/2026112749990735873)]?)

- Keep looking for a way to prompt an agent to get away from gradient descent 

        - (todo: Yaroslav) prototype binary prediction task, predicting next bit in text

# Ai summary 

### Participants and Backgrounds 

The Sutro Group is a volunteer-driven initiative composed of veteran AI researchers, hardware engineers, and domain outsiders united by the goal of finding energy-efficient, hardware-native alternatives to backpropagation.

- Yaroslav Bulatov (Host & Organizer): A 20+ year AI veteran who led the first deep learning deployment at Google and worked on gradient checkpointing at OpenAI. He famously beat Google in the 2018 DawnBench competition by optimizing infrastructure to reduce iteration times to 10 seconds. He is \"unretiring\" to disrupt Nvidia's software monopoly using AI agents.
- Michael Keating: Comes from a climate tech and electric mobility background. He is currently an executive at a data center cooling tech startup and strongly advised Yaroslav on project governance, suggesting long-term \"patient capital\" (like family offices or sovereign wealth funds) over traditional venture capital.
- Jon Belay: A South Park Commons member and former Google/Harvard researcher (Economic Fairness team). He runs an independent lab focused on deterministic methods for LLM pre-training and licenses algorithms to chip companies (like Nvidia and Etched) to solve NP-complete chip layout problems.
- Seth Stafford: A \"recovering mathematician\" (PhD from Cornell, 1991) and early Oracle employee. He currently applies AI to healthcare and is highly active in the group's technical execution, constantly running Claude Code agents to test hypotheses and verify the group's Python experiments.
- Uliana Popov: Handled the meeting's logistics and ordered the pizza. She is connected to the group through the local Russian community.
- Eric Frank (Left early): 
- Ritankar Das (Left early): A highly successful billionaire entrepreneur with an exit in biomedical advanced research.
- Isaac Rehg: Works in the hardware space and is currently with stealh AI startup building specialized inference chips.
- Andy Zhang
- (Note: Joshua Marks and Daria Soboleva \[Cerebras\ were absent. Remote asynchronous contributors like ]Emmett Bicker, and Germain Brion were heavily featured during the presentation).

------------------------------------------------------------------------

### Meeting Summary (Following the Speaker Notes) 

0. Logistics, Google Docs, Telegram Yaroslav opened by establishing the Google Calendar invite as the central hub for the group, linking to past notes, Colab notebooks, and the Telegram chat. A brief debate occurred regarding moving from Telegram to Discord or Slack to allow for better message threading, though Telegram's reliability currently wins out.

1. Motivation, 20-Year Plan. Why Unretire Now? Yaroslav shared his core thesis: deep learning algorithms like gradient descent were invented for CPUs and awkwardly retrofitted onto highly parallel GPUs 15 years ago. Backpropagation is inherently sequential and wastes massive amounts of energy accessing global memory. Yaroslav is becoming active now because AI coding agents allow a single researcher to prototype 100x faster. His ultimate goal is to break Nvidia's \$5 trillion software moat (CUDA) by using AI agents to instantly generate customized learning ecosystems for any hardware.

2. Last 6 Meetings Recap Yaroslav recapped the group's journey: from exploring Geoffrey Hinton's \"Forward-Forward\" algorithm (which Jamie Simon implemented via AI in 10 minutes), to practicing physical GPU power measurement (Joules) via Colab, to establishing the MicroGPT and Karpathy's makemore character-prediction benchmarks.

3. Existing Results, Overview The group reviewed recent rapid prototyping wins from the community:

- Germain Brion: Despite having a non-technical sales background, Germain used Claude to test \"truncated backpropagation.\" By dropping backward passes in early layers, he reduced energy costs by 19% and improved \"intelligence-per-joule\" by 27% without hurting validation loss.
- Emmett Bicker: Used an autonomous agentic loop (\"Esther\") to successfully reduce the memory footprint of MicroGPT from 80MB to 35MB.
- Andy Zhang: Used GLM-5 to optimize a 3-character prediction task, dropping energy use drastically (though the model \"cheated\" by finding a degenerate shortcut and memorizing n-grams).

4. Lessons - Iteration is Too Hard Despite these wins, Yaroslav highlighted a critical roadblock: the MicroGPT benchmark takes 3 minutes to train. For an AI agent to truly invent novel mathematical paradigms, it needs to iterate thousands of times. \"It feels like we're really missing a small task we can rapidly iterate on, like a simple drosophila (fruit fly) of learning, which could take one second to train.\"

5. Pebbling Games To bridge the gap between theoretical math and hardware realities, Yaroslav introduced the \"Pebbling Game\" (from compiler theory). On an H100 GPU, doing math is virtually free (0.3 picojoules), accessing local registers is cheap (5 picojoules), but moving data from global HBM memory is massively expensive (500+ picojoules). Backpropagation is inefficient because it requires storing activations across layers in HBM. Yaroslav proposed framing ML training as a scheduling game where AI agents must treat data as \"pebbles\" and are heavily penalized for using expensive HBM pebbles.

Socializing & Discussion The meeting transitioned into pizza and a deep discussion on project funding. Yaroslav expressed anxiety about taking Ritankar's \$500M VC pitch, noting that VCs force startups to optimize for standard SaaS metrics (\"building faster horses\") rather than paradigm shifts (\"burning down the forest\"). Michael Keating strongly advised targeting Sovereign Wealth funds or mission-driven Family Offices, which have 10-to-20-year time horizons and don't require extractive VC exits.

------------------------------------------------------------------------

### Streams of Work & Current Status 

1. Algorithmic Sanity Checks (Lead: Germain Brion)

- Status: Paused/Promising. Germain successfully proved that dropping backward passes in early layers saves energy and prevents overfitting on small models.
- Where it is: Paused pending a new hypothesis; needs to be tested on more complex datasets that resist simple memorization.

2. Agentic Pipeline Optimization (Leads: Andy Zhang, Emmett Bicker, Seth Stafford)

- Status: Active but bottlenecked. They are successfully using Claude Code and GLM-5 to automatically edit Python training scripts and measure Joules natively via CodeCarbon.
- Where it is: The agents are finding memory/schedule optimizations but take too long (3+ minutes per run) and occasionally \"cheat\" the learning task by bypassing deep learning entirely.

3. The Pebbling Game Simulator (Lead: Yaroslav Bulatov)

- Status: Early Prototyping. Yaroslav has mapped the H100 memory hierarchy costs, and Andy Zhang built a prototype \"mini-game\" scheduler in a Colab notebook.
- Where it is: Currently, the simulator only optimizes the memory schedule of existing algorithms. It needs to be updated to co-optimize the algorithm itself.

------------------------------------------------------------------------

### Suggestions for Most Promising Next Steps 

To achieve the ultimate goal of creating a process to instantly generate custom learning ecosystems for any hardware, the project must solve the iteration speed bottleneck.

1. Define the 1-Second \"Drosophila\" Task (Immediate Priority) The project is paralyzed by the 3-minute MicroGPT training loop. The group must abandon NLP token prediction to find a sub-1-second learning task.

- Suggestion: Implement a Binary Stream Prediction task (e.g., predicting the next bit of a compressed Wikipedia byte-stream) or a continuous XOR logic learning task. Alternatively, adopt the 191-parameter micro-Transformer mentioned by Andy. This strips away tokenization and embedding overhead, allowing agents to run hundreds of architectural experiments per minute.

2. Joint Optimization via the Pebbling Game Currently, agents are confused about whether they are optimizing a memory schedule or writing new math.

- Suggestion: Hardcode the Red-Blue Pebble Game constraints directly into a custom evaluation loop (Registers = 1 cost point, HBM = 100 cost points). Prompt the agent: \"Do not use PyTorch autograd. Invent a localized 'message-passing' algorithm that solves the 1-second Drosophila task while keeping all data inside Registers.\" This forces the LLM to organically invent biology-like, localized update rules natively.

3. Standardize the \"Hardware-Agnostic\" Sandbox Combine Emmett's agentic loop with Andy's CodeCarbon profiling into a standardized API. The pipeline should take three inputs: 1) A target hardware profile (latency/energy per memory tier), 2) The 1-second Drosophila task, and 3) An accuracy threshold. Once the agent can output an energy-efficient algorithm for an Nvidia H100 profile, that exact pipeline can be pointed at an AMD or Taalas chip tomorrow.

4. Secure \"Patient\" Capital Heed Michael Keating's advice. Position Sutro as an \"Energy/Climate AI\" initiative. Pitching to family offices or climate funds focused on reducing AI's gigawatt-hour footprint will yield patient capital that supports a \"blank slate\" open-source vision, completely avoiding the pressure to conform to the Nvidia/VC status quo.
