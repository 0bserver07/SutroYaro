!!! info "Cross-references"
    **Source**: [Google Doc](https://docs.google.com/document/d/1IdXRUhPRoWt8xLH1Y6iRWRx1g9-gbotFiiAnVixJYZY/edit?tab=t.0) · [Meeting #2 summary](../meetings/notes.md#meeting-2-26-jan-26-forward-forward-algorithm)
    **Experiment results**: [Exp E - Forward-Forward](../findings/exp_e_forward_forward.md) — FF has 25x worse ARD than backprop for 2-layer networks
    **Related**: [Jamie Simon's implementation](https://docs.google.com/document/d/1u8pIWg2iWc9R-dQgXz2Yt6xgF2fWP0GbDipbIxF1v1Y/edit?tab=t.0)

This is a detailed summary of the reading group discussion regarding Geoffrey Hinton's \"The Forward-Forward Algorithm: Some Preliminary Investigations.\"

This summary synthesizes the paper's core claims with the common critiques, questions, and follow-up topics that typically arise in technical deep dives on this work.

------------------------------------------------------------------------

### 1. Detailed Summary of the Paper 

Title: The Forward-Forward Algorithm: Some Preliminary Investigations (Hinton, 2022)

Core Motivation:

The paper proposes an alternative to Backpropagation (BP). Hinton argues that BP is flawed as a model of biological learning and effectively blocks the development of energy-efficient analog hardware.

+1

- Bio-Implausibility: The brain cannot \"freeze\" neural activity to propagate error derivatives backward, nor does it have the symmetric connectivity required to send errors back through the same synapses used for the forward pass.\
  
- Hardware Constraints: BP requires perfect knowledge of the forward pass to compute derivatives. This prevents the use of analog or \"black box\" hardware (which is variable and noisy) and prevents \"pipelining\" (processing a continuous stream of data) because the network must wait for the backward pass.\
  

The Mechanism (Forward-Forward):

Instead of a forward pass followed by a backward pass, the algorithm uses two forward passes with opposing objectives.

- The Positive Pass (Wake): The network processes \"real\" data (e.g., an image of a 7 combined with the label 7). The objective is to maximize \"Goodness\" (neural activity) in every layer.\
  
- The Negative Pass (Sleep): The network processes \"negative\" or \"hallucinated\" data (e.g., a hybrid image of a 7 and a 1, or an image with a wrong label). The objective is to minimize \"Goodness\".\
  

Key Technical Details:

- Goodness Function: Typically defined as the sum of squared neural activities (ReLU outputs) in a layer.\
  
- Greedy Layer-wise Learning: Each layer has its own objective function and updates its weights immediately, independent of the layers above or below it. This removes the need for a global error signal.\
  +1\
  
- Normalization: To prevent activity from simply exploding to maximize goodness, the input vector to each layer is length-normalized. This forces the layer to learn from the orientation (spatial pattern) of the activity, not the magnitude.\
  
- Inference: For classification, the label is usually embedded into the input (e.g., a one-hot code in the corner). At inference, the network runs multiple passes---one for each possible label---and selects the label that accumulates the highest \"goodness\" score.\
  +1\
  

------------------------------------------------------------------------

### 2. Summarized Questions and Concerns 

During the discussion, the group likely identified several \"sticking points\" where the algorithm faces theoretical or practical hurdles:

#### A. The \"Negative Data\" Problem 

- Question: How do we generate valid \"negative data\" for complex tasks?
- Concern: In the paper, Hinton uses spatial masks to stitch together MNIST digits. The group likely noted that generating high-quality negative data (that lies near the real data manifold but is clearly incorrect) is non-trivial for domains like NLP or Video. If the negative data is too easy (e.g., white noise), the network learns nothing; if it is too hard, the network fails to converge.

#### B. The \"Greedy\" Limitation 

- Question: Does greedy, layer-wise training limit the depth of learning?
- Concern: Since Layer 1 optimizes its weights without feedback from Layer 10, it might discard information that seems useless locally but is critical for high-level abstraction. This contrasts with Backprop, which can credit-assign through depth.
- Critique: This effectively turns a Deep Net into a stack of shallow nets. Early experiments show FF struggles to match Backprop on datasets much complex than MNIST (like CIFAR-10) without specific hacks.

#### C. Inference Efficiency 

- Question: Is this actually slower than Backprop?
- Concern: The inference method (running the forward pass \$N\$ times, once for each class label) scales poorly. For a 1000-class ImageNet task, this would be prohibitively slow compared to a single Softmax pass in standard networks.

#### D. Biological Plausibility vs. \"Hacks\" 

- Question: Is Layer Normalization biologically plausible?
- Concern: The algorithm relies heavily on normalizing activity vectors between layers to prevent goodness leakage. Some members may have argued that this is just as biologically suspect as Backprop, as it requires a global calculation over the layer.

------------------------------------------------------------------------

### 3. Follow-up Points & To-Dos 

Based on the paper and the discussion, here are recommended next steps to deepen your understanding:

Conceptual & Theoretical:

- Read \"Mortal Computation\": Review Hinton's broader philosophy on \"Mortal Computation.\" It explains why he wants algorithms that run on analog hardware (where weights cannot be copied), which is the true use-case for FF.
- Compare with Predictive Coding: FF is often compared to Predictive Coding (PC). Look into the differences: FF maximizes/minimizes \"Goodness\" (Energy), while PC typically minimizes \"Prediction Error.\"\
  +1\
  

Experimental (Coding Projects):

- Reproduce MNIST: Clone a simple PyTorch implementation.

<!-- -->

- Experiment: Visualize the \"Negative Data.\" Try changing the negative generation method (e.g., use random noise vs. permuted pixels) and observe how the accuracy collapses. This proves the algorithm relies entirely on the quality of the \"hallucinations.\"

<!-- -->

- The \"Sandwich\" Method: Address the inference speed concern by using FF to train the first 3 layers (as a feature extractor), freezing them, and then training a standard Linear Classifier (Softmax) on top using Backprop. This allows for \$O(1)\$ inference while still using FF for feature learning.
- Permutation Invariance Test: Flatten MNIST images and shuffle the pixels. Train an FF network on this. It often performs surprisingly well compared to ConvNets, highlighting that FF relies on global correlations rather than local geometry.

Advanced Reading:

- The Recurrent Extension: If the \"Greedy\" critique (Point B) was a major concern, read the specific section of the paper on Recurrent Forward-Forward. Hinton proposes treating static images as a video sequence, allowing later layers to send feedback to earlier layers over time, which theoretically solves the greedy limitation.\
