---
# try also 'default' to start simple
theme: default
layout: center
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
# background: https://cover.sli.dev
# some information about your slides, markdown enabled
title: Welcome to Slidev
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
# apply any unocss classes to the current slide
class: text-center
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# https://sli.dev/guide/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/guide/syntax#mdc-syntax
mdc: true
---

# Autoregressive Generation of Neural Field Weights

Using a transformer based architecture


<div class="h-8" />

<span class="italic op-[0.5]">from Luis Muschal and Luca Fanselau</span>

---

# TOC

- Introduction
- Related Works

- Regression Transformer
  - Structure Challenge (Permutation Invariance)
    - permutate the weights matrix
    - train with 1, 2, 4, 8, 16, 32 samples and see progress 
  - Token Problem (eg. SOS, EOS, Empty)
- Custom Overfitting
  - Abusing pretrained
  - Comparison Neural Fields (Unstructured vs Pretrained)
  - Animation Overfitting
- Token Prediction with Transformer (Classical Transformer)
  - Tokenization Technique (Naive, Bucket (Volume preserving), Learned)
- Outlook
  - Using Graph Structure to build better Tokenization
  

---
---
# Related works

<div class="grid grid-cols-2 gap-4">
  <div class="bg-white text-black p-4 rounded-md">
    <h2>HyperDiffusion</h2>
    <p>Generating neural implicit fields by predicting their weight parameters using diffusion</p>
  </div>
  <div class="bg-white text-black p-4 rounded-md">
    <h2>MeshGPT</h2>
    <p>Sequence-based approach to autoregressively generate triangle meshes as sequences of triangles</p>
  </div>
</div>


<!--
Hyperdiffusion:
- generating neural implicit fields by predicting their weight parameters using diffusion

MeshGPT:
- sequence-based approach to autoregressively generate triangle meshes as sequences of triangles
-->

---
---

# Neural Fields
Signal Compression and Signed Distance Function

Neural fields maps an input coordinate location in n-dimensional space to the target signal domain

Example:

With $S$ being a surface in a 3-dimensional space $\mathbb{R}^3$. The Signed Distance Function $f : \mathbb{R}^3 \rightarrow \mathbb{R}$ is defined for a point $\mathbf{p} \in \mathbb{R}^3$ as:


$$
f_{\Theta}(\mathbf{p}) = 
\begin{cases} 
\text{distance}(\mathbf{p}, S) & \text{if } \mathbf{p} \text{ is outside } S, \\
0 & \text{if } \mathbf{p} \text{ is on } S, \\
-\text{distance}(\mathbf{p}, S) & \text{if } \mathbf{p} \text{ is inside } S,
\end{cases}
$$

<!--
TODO: Examples of overfitted
-->


<!--
maybe without theta -> then clip -> boom theta appears
-->

<!--
Neural Fields (NeF): 
- Neural fields are continuous functions parameterized by neural network
- Neural fields maps an input coordinate location in n-dimensional space to the target signal domain
  - represent various types of spatial information, such as 3D geometry
- example: neural network encoding signed-distance function input (x, y, z) -> sdf-value 
  - from this the 3D-scene can be reconstructed (by sampling the space)
sdf:
  -positive values indicate points outside the surface
  -Negative values indicate points inside the surface.

Dunno if this is neccessary:
  Difference to Neural radiance fields (NeRF):
  - capturing both radiance (light emitted in different directions) and density.

 Neural Fields are encoded in the model weights -> our goal is to generate new MLPs that represent new structures in an autoregressive process
  latex code: P(X_t \mid X_{t-1}, X_{t-2}, \ldots, X_{t-p})
 
  using a transformer architecture -> parallel to chatGPT instead of generate the next word tokens we generate the next MLP-weight until we have a new MLP
 latex code: P(X_t \mid X_{t-1}, X_{t-2}, \ldots, X_{t-p})

question:

- implicit neural field - what would be an explicit neural field?
-->


---
---
# Autoregressive Generation of Neural Field Weights
And a regression transformer architecture

- Goal: generative modeling of neural fields
$P(\Theta_{i} \mid \Theta_{i-1}, \Theta_{i-2}, \ldots, \Theta_{0})$

- Using a generally available preset for GPT-like Architecture (like nanoGPT)
- Adapt to regression task

<video src="/autoregressive.mp4" autoplay loop muted></video>

<!--
Show history dependent process of autoregression

explain our approach:
- instead of predicting tokens predict weights directly

EG: Animation -> Single token in blackback -> two tokens -> more
-->

---
---

# Autoregressive Generation of Neural Field Weights
And a regression transformer architecture

<div class="grid grid-cols-2">
        <div><strong>nanoGPT</strong></div>
        <div><strong>our regression transformer</strong></div>
        <div>Token Embedding</div>
        <div>Weight to Embedding using MLP</div>
        <div>Embedding + Positional Encoding</div>
        <div>Embedding + Positional Encoding</div>
        <div>Nx Blocks (Causal Self Attention and MLP)</div>
        <div>Nx Blocks (Causal Self Attention and MLP)</div>
        <div>Linear Transformation Embedding</div>
        <div>Linear Transformation Embedding</div>
        <div>Softmax and Cross-Entropy Loss</div>
        <div>L1-norm as Loss</div>
</div>




<video src="/autoregressive.mp4" autoplay loop muted></video>

<!--
Show history dependent process of autoregression

explain our approach:
- instead of predicting tokens predict weights directly

EG: Animation -> Single token in blackback -> two tokens -> more
-->

---
---

# Regression Transformer 
Using neural fields that are randomly initialized

<VideoPane :rowLabels="['Ground Truth', 'N=4', 'N=32']" :videos="[['/regression_transformer/n_1_type_unconditioned_model_big_idx_0.mp4'], ['/regression_transformer/n_4_type_unconditioned_model_big_idx_0.mp4', '/regression_transformer/n_4_type_unconditioned_model_big_idx_1.mp4', '/regression_transformer/n_4_type_unconditioned_model_big_idx_2.mp4', '/regression_transformer/n_4_type_unconditioned_model_big_idx_3.mp4'],
['regression_transformer/n_32_type_unconditioned_model_big_idx_0.mp4','regression_transformer/n_32_type_unconditioned_model_big_idx_1.mp4','regression_transformer/n_32_type_unconditioned_model_big_idx_2.mp4','regression_transformer/n_32_type_unconditioned_model_big_idx_3.mp4']]" />

---
layout: two-cols
---


<template v-slot:default>

# Challenges: Permutation Symmetries
The same signal can be represented by different weight matrices

with $P$ being a permutation matrix and
$$
\~{W}_{1} = PW_{1} \\
\~{W}_{2} = W_{2}P^T \\
P^TP = \mathbf I 
$$


$$
\begin{aligned}
&f_{\~{W}}(x) = \~{W}_{2}\sigma(\~{W}_{1}x) = W_{2}P^T\sigma(P\~{W}_{1}x) \\
&=  W_{2}P^TP\sigma({W}_{1}x) = W_{2} \mathbf I W_{1}x = W_{2}\sigma(W_{1}x) = f_{W}(x)
\end{aligned}
$$

$$
P = \left[ 
\begin{array}{cccc}
0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0 \\
\end{array}\right]
$$

</template>
<template v-slot:right>

<video src="/nn-permutation.mp4" autoplay loop muted></video>

</template>

---
---

# Solution: Conditioned Training

Hypothesis: We can minimize the structure change of neural fields encoding similar signal by conditioning the training process in the initialization
Approach: Overfit Neural Fields using pretrained weight for initialization and random initialization


<!---
  - Text Introduction for training
-->

---
---

# Overfitting Neural Fields
Overfitting on one sample

- First start with ground truth and training of one initial sample
- Introduce weight visualization of weight matices bad biases

<VideoPane :rowLabels="['Ground Truth', 'N=4', 'N=32']" :videos="[]" />

---
---

# Overfitting Neural Fields
Comparison Conditiones and Unconditioned Training

- Introducing a new sample 35
- train two different NeFs one unconditiones and one conditioned
- structure might be similar but difficult to identify do to permutation symmetry

<VideoPane :rowLabels="['Ground Truth', 'N=4', 'N=32']" :videos="[]" />

---
---


# Overfitting Neural Fields
Using pretrained initialization

<VideoPane :rowLabels="['Ground Truth', 'N=4', 'N=32']" :videos="[]" />

---
---

# Overfitting Neural Fields - Comparison
How does the structure change while overfitting using pretrained?


$$
\begin{aligned}
\Delta(W) &= W_{\text{pretrained}} - W\\
\Delta(b) &= b_{\text{pretrained}} - b
\end{aligned}
$$
<VideoPane :rowLabels="['Ground Truth', 'N=4', 'N=32']" :videos="[]" />



---
---


# Regression Transformer (Conditioned Initialization)

<VideoPane :rowLabels="['Ground Truth', 'N=4', 'N=32']" :videos="[['/n_1_type_pretrained_model_big_idx_0.mp4'], ['/n_4_type_pretrained_model_big_idx_0.mp4', '/n_4_type_pretrained_model_big_idx_1.mp4', '/n_4_type_pretrained_model_big_idx_2.mp4', '/n_4_type_pretrained_model_big_idx_3.mp4']]" />

---
---

# Challenges: Tokenization
We run into issues regarding special tokens (what comes after the start token in the absence of the start token)




---
---

# Outlook: Token Prediction with Transformer


---
---
