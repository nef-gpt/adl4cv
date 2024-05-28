---
# try also 'default' to start simple
theme: default
layout: center
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
# background: https://cover.sli.dev
# some information about your slides, markdown enabled
title: Autoregressive Generation of Neural Field Weights
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
hideInToc: true
colorSchema: dark
---

# Autoregressive Generation of Neural Field Weights

Using a transformer based architecture


<div class="h-8" />

<span class="op-[0.5] text-sm">from Luis Muschal and Luca Fanselau</span>

---
hideInToc: true
---

# Table of Content

<Toc>
</Toc>

---
layout: flex
---
# Related works

<div class="grid grid-cols-2 gap-4 flex-1">
  <div class="bg-white text-black p-4 rounded-md flex flex-col max-h-100% justify-between items-center">
  <div>
    <h2>HyperDiffusion</h2>
    <p>Operates on MLP weights directly to generates new neural implicit fields encoded by synthesized MLP parameters</p>
    </div>
    <img src="/hd_overview.png" class="rounded-md w-350px object-contain" alt="HyperDiffusion">
    <span class="text-right w-100% text-gray-500 text-xs">
    ICCV’23 [Erkoç et al.]: Hyperdiffusion
    </span>
  </div>
  <div class="bg-white text-black p-4 rounded-md flex flex-col  max-h-100% justify-between items-center">
  <div>
    <h2>MeshGPT</h2>
    <p>Autoregressively generate triangle meshes as sequences of triangles using a learned vocabulary of latent quantized embedding as tokens</p>
    </div>
    <img src="/mesh_gpt_overview.png" class="rounded-md w-220px object-contain" alt="MeshGPT">
    <span class="text-right w-100% text-gray-500 text-xs">
    CVPR’24 [Siddiqui et al.]: MeshGPT
    </span>
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

# Autoregressive Generation of Neural Field Weights
Neural Fields

Input coordinate location in n-dimensional space are mapped to target signal domain

**Example:**

With $S$ being a surface in a 3-dimensional space $\mathbb{R}^3$. The Signed Distance Function $f : \mathbb{R}^3 \rightarrow \mathbb{R}$ is defined for a point $\mathbf{p} \in \mathbb{R}^3$ as:

<div class="flex flex-row gap-[2em] justify-between items-center">


$$
f_{\Theta}(\mathbf{p}) = 
\begin{cases} 
\text{distance}(\mathbf{p}, S) & \text{if } \mathbf{p} \text{ is outside } S, \\
0 & \text{if } \mathbf{p} \text{ is on } S, \\
-\text{distance}(\mathbf{p}, S) & \text{if } \mathbf{p} \text{ is inside } S,
\end{cases}
$$

<div class="rounded-4 bg-white p-4">
<video src="/hd_plane.mp4" width="200px" autoplay loop muted></video>
</div>

</div>

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
hideInToc: true
---
# Autoregressive Generation of Neural Field Weights
Autoregressive Process

- Goal: generative modeling of neural fields $P(\theta_{i} \mid \theta_{i-1}, \theta_{i-2}, \ldots, \theta_{0})$

- Using a generally available preset for GPT-like Architecture (like nanoGPT)
- Adapt to regression task $\theta_{i} =  \text{Transformer}(\theta_{i-1}, \theta_{i-2}, \ldots, \theta_{0})$

<video src="/autoregressive.mp4" autoplay loop muted></video>

<!--
Show history dependent process of autoregression

explain our approach:
- instead of predicting tokens predict weights directly

EG: Animation -> Single token in blackback -> two tokens -> more
-->

---
layout: flex
hideInToc: true
---

# Autoregressive Generation of Neural Field Weights
From nanoGPT to Regression Transformer



<VideoPane :rowLabels="['Ground Truth', 'N=1']" :videos="[['/regression_transformer/ground_truth_0.png'], ['/regression_transformer/n_1_type_unconditioned_model_big_idx_0.mp4']]" size="140px">
  <template v-slot:left-pane>
    <div class="grid grid-cols-[1fr_auto_1fr] gap-y-4px">
          <div class="border-b border-white"><strong>nanoGPT</strong></div>
          <div class="border-b border-white font-bold pr-4">vs.</div>
          <div class="border-b border-white"><strong>Our Regression Transformer</strong></div>
          <div class="text-#fde725">Token Embedding</div>
          <div></div>
          <div class="text-#fde725">Weight to Embedding using MLP</div>
          <div>Embedding + Positional Encoding</div>
          <div></div>
          <div>Embedding + Positional Encoding</div>
          <div>Nx Blocks (Causal Self Attention and MLP)</div>
          <div></div>
          <div>Nx Blocks (Causal Self Attention and MLP)</div>
          <div>Linear Transformation Embedding</div>
          <div></div>
          <div>Linear Transformation Embedding</div>
          <div class="text-#fde725">Softmax and Cross-Entropy Loss</div>
          <div></div>
          <div class="text-#fde725">L1-norm as Loss</div>
  </div>
  </template>


</VideoPane>

<!--
Show history dependent process of autoregression

explain our approach:
- instead of predicting tokens predict weights directly

EG: Animation -> Single token in blackback -> two tokens -> more
-->

---
layout: flex
---

# Regression Transformer 
Observing the Effects of Increasing N

<VideoPane :rowLabels="['Ground Truth', 'N=4', 'N=32']" :videos="[['/regression_transformer/ground_truth_0.png', '/regression_transformer/ground_truth_1.png', '/regression_transformer/ground_truth_2.png', '/regression_transformer/ground_truth_3.png'], ['/regression_transformer/n_4_type_unconditioned_model_big_idx_0.mp4', '/regression_transformer/n_4_type_unconditioned_model_big_idx_1.mp4', '/regression_transformer/n_4_type_unconditioned_model_big_idx_2.mp4', '/regression_transformer/n_4_type_unconditioned_model_big_idx_3.mp4'],
['regression_transformer/n_32_type_unconditioned_model_big_idx_0.mp4','regression_transformer/n_32_type_unconditioned_model_big_idx_1.mp4','regression_transformer/n_32_type_unconditioned_model_big_idx_2.mp4','regression_transformer/n_32_type_unconditioned_model_big_idx_3.mp4']]" size="90px">

<template v-slot:left-pane>

- Transformer fails to capture the structure of the weights for larger N
- Why can't the sequence be remembered even for small values of N?

</template>

</VideoPane>

---
layout: two-cols
---


<template v-slot:default>

# Challenges: Permutation Symmetries
The same signal can be represented by different weight matrices

$$
P = \left[ 
\begin{array}{cccc}
0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0 \\
\end{array}\right]
$$
permutated weight matrices are calculated using:
$$
\begin{aligned}
\~{W}_{0} &= PW_{0} \\
\~{W}_{1} &= W_{1}P^T \\
\end{aligned}
$$




</template>
<template v-slot:right>

<video src="/nn-permutation.mp4" autoplay loop muted></video>

</template>

---
layout: flex
transition: fade
---

# Overfitting Neural Fields
Finding a Solution

<!-- - First start with ground truth and training of one initial sample
- Introduce weight visualization of weight matices bad biases -->

<div class="flex-shrink-1 flex-grow-0 w-250px">

<div class="p-2 rounded-4 border border-white text-center text-sm">
Minimize structural change by conditioning the training process using weight initialization
</div>

Approach:
<div class="flex flex-col gap-[1em]">
<div class="p-2 rounded-4 border border-white text-center text-sm ">
Overfit single sample 
</div>
<div class="p-2 rounded-4 border border-white text-center text-sm ">
Use weights for different sample (conditioned)
</div>
<div class="p-2 rounded-4 border border-white text-center text-sm">
Train sample on randomly initialized weights (unconditioned)
</div>
</div>


</div>


---
layout: flex
transition: fade
---

# Overfitting Neural Fields
Overfitting on one sample

<!-- - First start with ground truth and training of one initial sample
- Introduce weight visualization of weight matices bad biases -->

<TrainingPane gt="/mnist_gt/mnist_0.png" :videos="['/comparison_0/unconditioned_0_cropped.mp4']" :labels="['First Sample']">

<template v-slot:left-pane>
<div class="flex-shrink-1 flex-grow-0 w-250px">

<div class="p-2 rounded-4 border border-white text-center text-sm">
Minimize structural change by conditioning the training process using weight initialization
</div>

Approach:
<div class="flex flex-col gap-[1em]">
<div class="p-2 rounded-4 border border-white text-center text-sm text-black bg-white">
Overfit single sample 
</div>
<div class="p-2 rounded-4 border border-white text-center text-smr text-sm ">
Use weights for different sample (conditioned)
</div>
<div class="p-2 rounded-4 border border-white text-center text-sm">
Train sample on randomly initialized weights (unconditioned)
</div>
</div>


</div>
</template>

</TrainingPane>

---
layout: flex
hideInToc: true
transition: fade
---

# Overfitting Neural Fields
Overfitting on other sample


<TrainingPane gt="/mnist_gt/mnist_35.png" :videos="['/comparison_11_35_47_65/unconditioned_35_cropped.mp4', '/comparison_11_35_47_65/pretrained_35_cropped.mp4']" :labels="['Unconditioned', 'Conditioned']" infoBox="/comparison_0/unconditioned_0_last_frame.png" infoLabel="Condition">

<template v-slot:left-pane>
<div class="flex-shrink-1 flex-grow-0 w-250px">

<div class="p-2 rounded-4 border border-white text-center text-sm">
Minimize structural change by conditioning the training process using weight initialization
</div>

Approach:
<div class="flex flex-col gap-[1em]">
<div class="p-2 rounded-4 border border-white text-center text-sm">
Overfit single sample 
</div>
<div class="p-2 rounded-4 border border-white text-center text-sm text-sm text-black bg-white">
Use weights for different sample (conditioned)
</div>
<div class="p-2 rounded-4 border border-white text-center text-sm text-black bg-white">
Train sample on randomly initialized weights (unconditioned)
</div>
</div>


</div>
</template>

</TrainingPane>

---
layout: flex
hideInToc: true
---

# Overfitting Neural Fields
Visualizing the Difference:


<TrainingPane gt="/mnist_gt/mnist_35.png" :videos="['/comparison_with_comparison_model_11_35_47_65/unconditioned_35_cropped.mp4', '/comparison_with_comparison_model_11_35_47_65/pretrained_35_cropped.mp4']" :labels="['Unconditioned', 'Conditioned']" infoBox="/comparison_0/unconditioned_0_last_frame.png" infoLabel="Condition">

<template v-slot:left-pane>
<div class="flex-shrink-1 flex-grow-0 w-250px h-100%">

<div class="flex flex-col gap-[1em] justify-center h-100%">
<div class="p-2 rounded-4 border border-white text-center text-sm">
$$
\begin{aligned}
\Delta(W) &= W_{\text{pretrained}} - W \\
\Delta(b) &= b_{\text{pretrained}} - b
\end{aligned}
$$
</div>


</div>




</div>
</template>

</TrainingPane>


---
layout: flex
transition: fade
---

# Reminder: Regression Transformer 
How far we got with unconditioned neural fields

<VideoPane :rowLabels="['Ground Truth', 'N=4', 'N=32']" :videos="[['/regression_transformer/ground_truth_0.png', '/regression_transformer/ground_truth_1.png', '/regression_transformer/ground_truth_2.png', '/regression_transformer/ground_truth_3.png'], ['/regression_transformer/n_4_type_unconditioned_model_big_idx_0.mp4', '/regression_transformer/n_4_type_unconditioned_model_big_idx_1.mp4', '/regression_transformer/n_4_type_unconditioned_model_big_idx_2.mp4', '/regression_transformer/n_4_type_unconditioned_model_big_idx_3.mp4'],
['regression_transformer/n_32_type_unconditioned_model_big_idx_0.mp4','regression_transformer/n_32_type_unconditioned_model_big_idx_1.mp4','regression_transformer/n_32_type_unconditioned_model_big_idx_2.mp4','regression_transformer/n_32_type_unconditioned_model_big_idx_3.mp4']]" size="90px">

<template v-slot:left-pane>

- Transformer fails to capture the structure of the weights for larger N
- Why can't the sequence be remembered even for small values of N?

</template>

</VideoPane>

---
layout: flex
---


# Regression Transformer
Using conditioned neural fields to verify the Hypothesis

<VideoPane :rowLabels="['Ground Truth', 'N=4', 'N=32']" :videos="[['/regression_transformer/ground_truth_0.png', '/regression_transformer/ground_truth_1.png', '/regression_transformer/ground_truth_2.png', '/regression_transformer/ground_truth_3.png'], ['/regression_transformer/n_4_type_pretrained_model_big_idx_0.mp4', '/regression_transformer/n_4_type_pretrained_model_big_idx_1.mp4', '/regression_transformer/n_4_type_pretrained_model_big_idx_2.mp4', '/regression_transformer/n_4_type_pretrained_model_big_idx_3.mp4'],
['regression_transformer/n_32_type_pretrained_model_big_idx_0.mp4','regression_transformer/n_32_type_pretrained_model_big_idx_1.mp4','regression_transformer/n_32_type_pretrained_model_big_idx_2.mp4','regression_transformer/n_32_type_pretrained_model_big_idx_3.mp4']]" size="90px">

<template v-slot:left-pane>

- Training Regression Transformer using conditioned Neural Fields Weights
- Structural similarity of weights improve the performance of the Transformer

</template>

</VideoPane>

---
layout: flex
---

# Outlook: Tokenization
Predicting the next MLP weight as a token

<div class="flex flex-col justify-center w-100% items-center">
<span class="text-justify">
We run into issues regarding special tokens (what comes after the start token in the absence of the start token)
</span>

$\theta_{i} =  \text{Transformer}(\theta_{i-1}, \theta_{i-2}, \ldots, \theta_{0}) \rightarrow \theta_{0}\text{?}$

<span class="text-justify">
Solution:
Find Tokens to encode the MLP weights and transfer from Regression Transformer to Classical Transformer
</span>
</div>
  

<div class="flex flex-row gap-[1em] justify-stretch items-stretch mt-4 flex-1">
<div class="p-4 rounded-4 border border-white text-center text-sm flex-basis-50% text-left">


## First Approach:
- Create Tokens using the Condition Neural Field Weights to train
- Naive Attempt: Define Buckets on different Metrices for quantization
- Vector Quantization Attempt: Find Tokens using Vector Quantization 


</div>
<div class="p-4 rounded-4 border border-white text-center text-sm flex-basis-50% text-left">


## Second Approach:
- Find vocabulary of latent quantized embeddings, using graph convolutions 
- Neural Networks are by nature acyclic graphs


</div>
</div>





---
layout: center
class: text-center
---

# Thank you for your attention!
We hope you enjoyed our presentation and are looking forward to your questions.

<div class="h-8" />
<span class="op-[0.5] max-w-30%">

If you want to access the slides afterwards, you can find them under: [https://adl4cv.vercel.app](https://adl4cv.vercel.app)

We run into issues regarding special tokens (what comes after the start token in the absence of the start token)

- $\theta_{i} =  \text{Transformer}(\theta_{i-1}, \theta_{i-2}, \ldots, \theta_{0})$ → $\theta_{0}\text{?}$
- due to the continuous nature of the regression transformer there are no special token (eg. SOS)

- Outlook
  - Using Graph Structure to build better Tokenization

</span>