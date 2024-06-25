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
- When transforming neural fields into sequences, the permutation problem arises

-->

---
---


# Experiment
From Regression Transformer to Traditional Transformer

<img src="/general-approach.png" alt="Training_Transformer">

**General Procedure**:
1. Tokenization of weights using Vector Quantization
2. Training Transformer with special tokens for start of sequence and conditioning
3. Optimizing Transformer Inference Parameters for novel neural fields from the same distribution

<!-- [Luca/Luis]
LUCA: TODO: Change Layout
-->

---
layout: two-cols-default
---


# Discretization
Tokenization of weights using Vector Quantization

::left::


**Approach**: Continuous Neural Field weights are discretized using Vector Quantization

**Procedure**:
- Learning Codebook using weights of all MNIST Neural Fields



::right::

<video src="/vq_1.mp4" autoplay loop muted></video>

<!-- [Luis]
- LUCA TODO: Center Text Left-hand-side

-->

---
layout: two-cols-default
---


# Discretization
Training of Vector Quantization

::left::

**Training**:
1. Codebook elements randomly initialized
2. Assign to the closest Codebook element
3. Update Codebook elements by minimizing L2-loss
4. Assigned rarely used elements to weights
5. goto 2.

::right::

<div class="p-2 rounded-[8px] border border-[#212121] bg-[black] shadow-xl">
<video class="w-80% mx-auto" src="/concatenated_images_animation.mp4" autoplay loop muted></video>
</div>


---
layout: two-cols-default
---

# Discretization
Special Tokens

::left::


**Special Tokens**:
- “Start of Sequence” Token **SOS**
  - indicating the start of the sequence
- “Conditioning” Token **C** 
  - indicating to which number the weights belong


::right::


<video src="/vq_2.mp4" autoplay loop muted></video>




<!-- [Luis]

TODO: Center DIV

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

TODO: Layout LATEX stuff

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

<span class="h-8 block" />

<img src="/mnist-metric.png" class="w-[100%] mx-auto my-[16px]" />

<!-- [Luca]

(Accuracy (argmax) and Cross Entropy Loss)

-->



---
layout: flex
---

# Training a Transformer
Hyperparameters

<style>

td {
  padding-bottom: 0.5em;
  padding-top: 0.5em;
}

th {
  padding-bottom: 0.5em;
  padding-top: 0.5em;
}

</style>

<div class="flex flex-row gap-4 items-center">
<div class="flex-1">

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
      <td style="text-align: right;">40000</td>
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
      <td style="text-align: right;">240</td>
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
     <tr>
      <td>Context Length</td>
      <td style="text-align: right;">562</td>
    </tr>
  </tbody>
</table>

</div>


<div class="flex-basis-60% flex-grow-0 pt-2em">

<img src="/plot_training.png" alt="Training_Transformer">

</div>
</div>

<!-- [Luis]
Table of Hyperparameters
Loss function for training and validation
MNIST Metrics
Pictures of Training Results (Just save pictures of MNIST Metrics)
TODO: Maybe bigger fond, lr two time
-->

---
layout: two-cols-default
---

# Preliminary Results
Autoregressive Generation and Initial Results

::left::

<video controls src="/inference.mp4" class="m-auto" loop muted></video>

::right:: 


<img v-click src="/transformer_naive.png" class="h-[80%] m-auto" alt="Training_Transformer">

<!-- [Luis]
- TODO: Maybe some more text

-->

---
layout: two-cols-default
---

# Tuning Inference Parameters
Determining top-k, temperaturen

::left::

- **Top-k**: Reduces number of considered tokens
- **Temperature**: Smooths the distribution of the logits

$$

L \hat{=} \text{Logits} \\
T \hat{=} \text{Temperature} \\
\text{Softmax}(L) = \frac{\exp(L/T)}{\sum_{i}\exp(L_i/T)} \\

$$

::right::

<div class="p-2 rounded-[8px] border border-[#212121] bg-[black] shadow-xl">
<video src="/animated_bar_chart.mp4" autoplay loop muted></video>
</div>

<!-- [Luca]
- Input to the generation procedure -> top-k temperature


- Performed a grid search over the hyperparameters top-k and temperature
- Combination of metrics

TODO: Latex layout / Layout Slides
TODO: Elaborate on temperature
-->



---
layout: flex
---

# Results
For all conditioning tokens


<div class="flex flex-row items-center h-100% gap-4">
<div class="flex-1 flex-basis-0px relative">
<div class="absolute left-0 top-0 bottom-0 w-93px bg-[#333] -z-10 rounded-md"></div>
<img src="/nn-plot-final-0-4.png" alt="Results for all conditioning tokens" class="" />
</div>
<div class="flex-1 flex-basis-0px relative">
<div class="absolute left-0 top-0 bottom-0 w-93px bg-[#333] -z-10 rounded-md"></div>
<img src="/nn-plot-final-5-9.png" alt="Results for all conditioning tokens" class="" />
</div>
</div>
<div class="flex flex-row justify-center">
<span>
  <span class="text-[#777] font-size-3">

  Results for all conditioning tokens for $T=0.8$ and $\text{top-k}=3$
  
  </span>
</span>
</div>

<!--  [Luca]

- Results for all conditioning tokens with nearest neighbor
- Achieved metrics for best configuration (?)

- TODO: Metrics for presented results

-->

---
---

# Outlook & Further Work
From MNIST to ShapeNet

<div class="relative">
<img v-click.hide src="/general-approach.png" alt="Training_Transformer">
<img v-after src="/general-approach-shapenet.png" alt="Training_Transformer" class="absolute top-0 bottom-0 left-0 right-0">
</div>

**Challenges**: 
- Neural Fields for ShapeNet have an increased complexity
- Context Length would be too high if all weights are used

**Solution**:
- Add global position using Sinusoidal Position Encoding

**Future**: 
- Retraining Transformer and perform Grid Search for optimal parameters
- Qualitative comparison to State-of-the-Art methods

<!-- [Luis]

TODO: Rework text

-->

---
layout: center
class: text-center
hideInToc: true
---
# Thank you for your attention!
We hope you enjoyed our presentation and are looking forward to your questions.

<div class="h-8" />
<span class="op-[0.5] max-w-30%">
</span>

<!-- [Luis Muschal]

-->