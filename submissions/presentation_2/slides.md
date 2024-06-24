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
layout: two-cols
---

# Recap
Weight initialization for neural fields

- **Goal**: Generate novel neural fields using an autoregressive process
- **Problem**: Permutation problem arises when transforming neural fields into sequences
- **Solution**: Condition the weights to decrease structural differences between neural fields

- **First presentation**: Trained a Regression Transformer to generate neural fields, but ran into novelity issues

<template v-slot:right>

<video src="/nn-permutation.mp4" autoplay loop muted></video>

</template>

<!-- [Luca]
TODO: reuse visualization from other presentation
- When transforming neural fields into sequences, the permutation problem arises

-->

---
---

# Experiment
From Regression Transformer to Traditional Transformer

General Procedure:
- Tokenization of weights using Vector Quantization
- Training Transformer with special tokens for start of sequence and conditioning
- Optimizing Transformer Inference Parameters for novel neural fields from the same distribution

<!-- [Luis]
same distribution of MNIST digits
-->

---
layout: two-cols
---


# Discretization
Tokenization of weights using Vector Quantization

**Approach**: Continuous Neural Field weights are discretized using Vector Quantization

**Procedure**:
- Learning Codebook using weights of all MNIST Neural Fields

**Training**:
1. Codebook elements randomly initialized
2. Assign to the closest Codebook element
3. Update Codebook elements using L2-loss
4. Assigned rarely used elements to weights
5. goto 2.

<template v-slot:right>



</template>


<!-- [Luis]

-->

---
---




# Discretization
Special Tokens


**Spezial Tokens**:
- “Start of Sequence” Token **SOS**
  - indicating the start of the sequence
- “Conditioning” Token **C** 
  - indicating which to which number weights belong

<!-- [Luis]

TODO: Create animation

-->

---
layout: two-cols-default
transition: fade
---


# Metrics
Introduction

::left::

<div class="p-4 border border-[#333] rounded-[8px] bg-[#181818]">

- $S_g$: Set of **evaluated** generated neural fields
  - Images generated from novel neural fields
- $S_r$: Set of **evaluated** reference neural fields
  - Images generated from training neural fields
- $D(X,Y)$: Distance between elements $X, Y \in S_g \cup S_r$
  - Here Structural Similarity Index (SSIM) is used

</div>
<!-- [Luca]

Structural Similarity Index (SSIM) - a metric to measure the similarity between two images
 works by comparing the similarity of the luminance, contrast, and structure of two images
 two windows of the images are compared, and the SSIM is calculated based on the comparison of the windows
 each window is compared in terms of mean, variance, and covariance
-->


---
layout: two-cols-default
transition: fade
---

# Metrics
Minimum Matching Distance (MMD)

::left::

<div class="p-4 border border-[#333] rounded-[8px] bg-[#181818]">

- $S_g$: Set of **evaluated** generated neural fields
  - Images generated from novel neural fields
- $S_r$: Set of **evaluated** reference neural fields
  - Images generated from training neural fields
- $D(X,Y)$: Distance between elements $X, Y \in S_g \cup S_r$
  - Here Structural Similarity Index (SSIM) is used

</div>

::right::

$$
\mathrm{MMD}(S_g,S_r) =\frac{1}{|S_{r}|}\sum_{Y\in S_{r}}\min_{X\in S_{g}}D(X,Y)
$$


<span class="h-8" />

- Average minimum distance for each element in $S_r$ to $S_g$
- Lower is better




<!-- [Luca]
Is there a close match in the generated set for each reference image?
-->

---
layout: two-cols-default
transition: fade
---


# Metrics
Coverage

::left::

<div class="p-4 border border-[#333] rounded-[8px] bg-[#181818]">

- $S_g$: Set of **evaluated** generated neural fields
  - Images generated from novel neural fields
- $S_r$: Set of **evaluated** reference neural fields
  - Images generated from training neural fields
- $D(X,Y)$: Distance between elements $X, Y \in S_g \cup S_r$
  - Here Structural Similarity Index (SSIM) is used

</div>


::right::


$$
\mathrm{COV}(S_g,S_r) =\frac{|\{\arg\min_{Y\in S_r}D(X,Y)|X\in S_g\}|}{|S_r|}
$$

<span class="h-8" />

- Percentage of reference images that are a closest neighbor in the generated set
- Amount of reference images that are covered by the generated set
- Higher is better



---
layout: two-cols-default
---


# Metrics
1-Nearest Neighbor Accuracy (1-NNA)

::left::

<div class="p-4 border border-[#333] rounded-[8px] bg-[#181818]">

- $S_g$: Set of **evaluated** generated neural fields
  - Images generated from novel neural fields
- $S_r$: Set of **evaluated** reference neural fields
  - Images generated from training neural fields
- $D(X,Y)$: Distance between elements $X, Y \in S_g \cup S_r$
  - Here Structural Similarity Index (SSIM) is used

</div>

::right::

$$
\begin{gather*}
1-\mathrm{NNA}(S_g,S_r) = \\ 
\frac{\sum_{X\in S_g}\mathbb{1}[N_X\in S_g]+\sum_{Y\in S_r}\mathbb{1}[N_Y\in S_r]}{|S_g|+|S_r|}
\end{gather*}
$$

<span class="h-8" />


- 50% is the optimal value
- Sum of the elements of $S_g$ and $S_r$ that are closest neighbors in their respective sets
- Divided by the total number of elements in $S_g$ and $S_r$




<!--  [Luca]
-->

---
---


# Metrics
MNIST Classifier Score

**Proxy metric**: Generate neural fields which lead to *understandable* digits

**Procedure**: 
- Train a classifier on MNIST dataset
- Generate a novel neural field using a conditioning token 
- Use the data pair (neural field, digit) to get a score from the classifier

<img src="/mnist-metric.png" class="w-[100%] mx-auto my-[16px]" />

<!-- [Luca]

(Accuracy (argmax) and Cross Entropy Loss)

-->



---
layout: two-cols-header
---

# Training a Transformer Model
Hyperparameters

::left::

<table class="font-size-3">
  <thead>
    <tr>
      <th><strong>Hyperparameter Training</strong></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Learning Rate</td>
      <td style="text-align: right">3e-3</td>
    </tr>
    <tr>
      <td>Iterations</td>
      <td style="text-align: right;">30000</td>
    </tr>
    <tr>
      <td>Batch Size</td>
      <td style="text-align: right;">64</td>
    </tr>
  </tbody>
</table>



<table class="font-size-3">
  <thead>
    <tr>
      <th><strong>Hyperparameter Transformer</strong></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Embedding Size</td>
      <td style="text-align: right;">64</td>
    </tr>
    <tr>
      <td>Numbers of Heads</td>
      <td style="text-align: right;">12</td>
    </tr>
    <tr>
      <td>Numbers of Attention Blocks</td>
      <td style="text-align: right;">12</td>
    </tr>
    <tr>
      <td>Vocabulary Size</td>
      <td style="text-align: right;">256</td>
    </tr>
  </tbody>
</table>


::right::

image content


<!-- [Luis]
Table of Hyperparameters
Loss function for training and validation
MNIST Metrics
Pictures of Training Results (Just save pictures of MNIST Metrics)
TODO:
loss function, MNIST metrics, training result
-->

---
---

# Tuning Inference Parameters
Determining top-k, temperature and top-p

<!-- [Luca]
-->

---
---

# Outlook & Further Work

- maybe show first working prototype

<!-- [Luis]
-->