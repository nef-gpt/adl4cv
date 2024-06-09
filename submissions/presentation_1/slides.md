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

<!-- [Luis Muschal]
hello welcome

I'm Luis this is Luca

our project is about:

autoregressive generation of Nerual field weights using a transformer based architecture

progress of our work

-->


---
hideInToc: true
---

# Table of Content

<Toc>
</Toc>

<!-- [Luca Fanselau]
0:00-1:00

Luis is going to start of by presenting mainly the two papers that inspired our work.

- Afterwards we will give a brief introduction to neural fields and the autoregressive process, so that we are all on the same page.e

- Then I will start with the first part of our work, the regression transformer, and how we adapted the transformer architecture to predict the weights of a neural field.

- But this first approach also comes along with some challenges, one of them being the permutation symmetries of the weights. Luis will explain what this is and how we tackled this issue.

- In the end, I will again show how we used these results to improve the performance of the regression transformer architecture.

- And finally, we will give a brief outlook on how we plan to further improve on the pipeline to enable autoregressive prediction of novel neural fields. 

-->


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

<!-- [Luis Muschal]
1:00
1:45

Mainly two different inspirations for our work


Hyperdiffusion 

Operates on MLP weights directly to generates new neural implicit fields encoded by synthesized MLP parameters

-> a transformer based architecture is used denoising

and

MeshGPT:
Autoregressively generate triangle meshes as sequences of triangles using a learned vocabulary of latent quantized embedding as tokens<


which we also learned about in the last lecture

uses vocabulary that is learned using graph convolutions

Our project is basically the combination of both worlds

-> we want to autoregressively generate new Neural Fields 

Now to make sure we are all on the same page some information on Neural Fields

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

<!-- [Luis Muschal]
1:45
2:30

Now to make sure we are all on the same page some information on Neural Fields

Neural Fields are used to map input coordinate location in n-dimensional space to a target signal domain

Example:
Lets say we have a surface in R^3 we can define a distance function 
This distance function can be learned using a neural network 
For this we randomly sample 3D points from our target as ground truth and train a neural network to memorize them



-->


---
hideInToc: true
---
# Autoregressive Generation of Neural Field Weights
Autoregressive Process

- Goal: generative modeling of neural fields $P(\theta_{i} \mid \theta_{i-1}, \theta_{i-2}, \ldots, \theta_{0})$

- Using a generally available preset for GPT-like Architecture (like nanoGPT)
- Use Transformer to sample from the Probability, eg. $\theta_{i} =  \text{Transformer}(\theta_{i-1}, \theta_{i-2}, \ldots, \theta_{0})$ 

<video src="/autoregressive.mp4" autoplay loop muted></video>

<!-- [Luca Fanselau]
2:30
3:00

What is the autoregressive process?

So generally the goal is to predict sequences of any kind by using the previous tokens to predict the next one. 

Since the output sequence is fed back into the model the input sequence and output sequence have the same domain. Therefore, architecture stypically drop the encoder part of the transformer and only use the decoder part. (as with openly available GPT models such as nanoGPT)

The Idea is that the model learns the distribution of the next token given the previous tokens and can sample from this distribution to generate new sequences.

We just have the problem that we don't have tokens but weights of a neural fields. So we need to adapt the transformer architecture to predict the weights of a neural field.

-->

---
layout: flex
hideInToc: true
---

# Autoregressive Generation of Neural Field Weights
From nanoGPT to Regression Transformer

<VideoPane :rowLabels="['Ground Truth', 'N=1']" :videos="[['/regression_transformer/ground_truth_0.png'], ['/regression_transformer/n_1_type_unconditioned_model_big_idx_0.mp4']]" size="140px">
  <template v-slot:left-pane>
  <div class="w-100% h-100% flex flex-col justify-center items-center">
    <div class="grid grid-cols-[1fr_auto_1fr] gap-y-8px text-center items-center">
          <div class="border-b border-white"><strong>nanoGPT</strong></div>
          <div class="border-b border-white font-bold pr-4">vs.</div>
          <div class="border-b border-white"><strong>Our Regression Transformer</strong></div>
          <div class="text-#fde725">Tokenizer</div>
          <div></div>
          <div class="text-#fde725">MLP Embedding on weight</div>
          <div class="grid-col-span-3 text-center">Embedding + Positional Embedding</div>
          <div class="grid-col-span-3 text-center">N x Blocks (Causal Self Attention and MLP)</div>
          <div class="grid-col-span-3 text-center">Linear Transformation Embedding</div>
          <div class="text-#fde725">Cross-Entropy Loss</div>
          <div></div>
          <div class="text-#fde725">L1-norm as Loss</div>
  </div>
  </div>
  </template>


</VideoPane>

<!-- [Luca Fanselau]
3:00
4:15

The regression transformer that we developed is based on nanoGPT, with some core differences.

nanoGPT is a decoder-only transformer architecture inspired by GPT-2.
 It consists of a Tokenizer, to convert the input sequence into tokens, 
 an Embedding layer, to convert the tokens into a multi dimensional embedding, 
 and a stack of N blocks, each containing a Causal Self Attention layer and a MLP layer. 
 
 The output of the last block is then transformed into a probability distribution over the tokens using a linear transformation and a softmax layer. The model is trained using cross-entropy loss.

Our regression transformer in contrast to nanoGPT uses the continous weights of the neural fields directly using an Embedding that maps the single value of the weight into a multi-dimensional embedding.

Additionally we changed the output layer to predict a single weight directly using a L1-norm loss function.

On the right side you can see how the architecture is overfitted on a single training sample. The output that you can see is predicted using an autoreressive process, where the model predicts the next weight based on the previous weights. The only input that the model gets is the first weight of the neural field.

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

<!-- [Luca Fanselau]
4:15
5:00

But once we scale up the training to more than one weight, the model fails to capture the structure of the weights for larger N. The model is not able to remember the sequence of weights even for small values of N like 32.

-->

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

<!-- [Luis Muschal]
5:00
5:45

issue -> certain permutations of parameters in neural networks do not change the underlying function

example -> permutation matrix like this on

use this formula -> function represented by the MLP stays the same

example -> 4*3*2*1 possibilities to permutate the layer without changing the underlying function

How can we reduce this effect

-->
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

<!--  [Luis Muschal]
5:45-6:15

Now train neural fields

hypotheses: minimize structural change by conditioning the training process using weight initialization

general idea:

1. overfit one mnist samlpe
2. use the pretrained weights of that sample for the weight initialization of another NeF
3. Proof of concept also train that same sample on randomly initialized weights


-->



---
layout: flex
transition: fade
hideInToc: true
---

# Overfitting Neural Fields
Overfitting on one sample

<!-- - First start with ground truth and training of one initial sample
- Introduce weight visualization of weight matices bad biases -->

<TrainingPane gt="/mnist_gt/mnist_0.png" :videos="['/comparison_0/unconditioned_0_cropped.mp4']" :labels="['First Sample']" infoBox="/comparison_11_35_47_65/unconditioned_65_last_frame.png" infoLabel="Legend">

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


<!-- [Luis Muschal]
6:15-6:45

First point -> fit one 

explain legend -> first matrix 16x18 (2 (x, y) -> 18 using positional encoding (not learned) -> hidden layer 16x16 -> projected to output 1x16


we see the weights an biases to encode the neural field

so in the beginning of the training the weights are just random

on the left we see the ground truth data

press button 

see the weight changing and the NeF getting closer to the ground truth


-->

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

<!-- [Luis Muschal]
6:45-7:30

different mnist sample

Now we come to the second step

two different weight initialization for the neural fields

left ranfomly initialized -> we see noisy image -> nothing learned yet
right we see the Neural Field which is initialized using the weights of the previous sample

maybe permutation could lead to the same picture


-->
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

<!-- [Luis Muschal]
7:30-8:00

it is also interesting to look at the difference between the pretrained weights and biases matrices and the trained conditioned
 weights and biases matrices

on the right we can see the change in the beginning is zero while the matrices on the left already have a large structural difference

while the MLPs encode the same image we see a large difference in the matrices elements

our conclusion is that we can verfy our hypotheses and hopefully have a better chance at training the regression transformer
-->

---
layout: flex
transition: fade
hideInToc: true
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

<!-- [Luca Fanselau]
8:00

So now after this digression, we want to come back to the regression transformer and test our hypothesis, that conditioning the training process using weight initialization can improve the performance of the transformer.

Right now we are looking at the exact slide that we have seen before, just to remind you how the model behaved when using unconditioned neural fields.

-->

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

<!-- [Luca Fanselau]
9:00

For this slide we now switched over to overfitting to neural fields that where initialized using another neural field. Let's see how the model behaves when using conditioned neural fields.

 WAIT

As you can see the same autoregressive process that we used before now leads to a much better reconstruction of the ground truth. The model is now able to remember the sequence of weights even for larger values of N like 32.

-->

---
layout: flex
---

# Conclusion and Outlook: Tokenization
Predicting the next MLP weight as a token

<div class="flex flex-col justify-center w-100% items-start">
<span class="text-justify">
Run into issues regarding special tokens:
</span>

<div class="w-100% flex flex-row justify-center">

$\theta_{i} =  \text{Transformer}(\theta_{i-1}, \theta_{i-2}, \ldots, \theta_{0}) \rightarrow \theta_{0}\text{?}$

</div>

<span class="text-justify">


**Solution:**
Find Tokens to encode the MLP weights and transfer from Regression Transformer to Classical Transformer Architectures


</span>
</div>
  

<div class="flex flex-row gap-[1em] justify-stretch items-stretch mt-4 flex-1">
<div class="p-4 rounded-4 border border-white text-sm flex-basis-50% text-justify">


<h3 class="text-center mb-2">First Approach:</h3>


- Create Tokens using conditioned Neural Field Weights
- Naive Attempt: Use weight distribution for discretization
- Vector Quantization Attempt: Find optimal token representation using optimization techniques


</div>
<div class="p-4 rounded-4 border border-white text-sm flex-basis-50% text-justify">

<h3 class="text-center mb-2">Second Approach:</h3>

- Find layer representations of unconditioned neural fields that are permutation equivariant
- For example by using the graph structure of the neural fields and employing deep learning techniques suited for graphs



</div>
</div>

<!-- [Luca Fanselau]
9:00
10:00

But of course we are not done yet. 

To use the autoregressivion to generate completely novel neural fields, we need to have what are called special tokens. Here you add tokens which have spacial meaning in the context of the sequence to the generation process. For Example the SOS or EOS token. To completely unconditionally create neural fields we need those tokens and therefore also a way to discretize our data.

For this we have two approaches in mind. 

The first one is to discretize the conditional neural field weights and use them directly as tokens in the sequence.

Naive Attemplt: Use all available data to build buckets that align with the distribution of the weights.
Vector Quantization: Find optimal token representation using optimization techniques

The second approach is to redefine the scope of a single token. Instead of using the weights of the field as the entity we want to find latent representations of the whole layer that don't experience the challenges with permutation symmetries. For example by using the graph structure of the neural fields and employing deep learning techniques suited for graphs to predict discrete tokens in an autoencoder like architecture.

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