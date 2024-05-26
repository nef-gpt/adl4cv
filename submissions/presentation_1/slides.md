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
# Introduction

- Predict Neural Field Weights as sequence

- What are Neural Fields?
- What is autoregressive generation? (maybe with probability) $\sqrt{3x-1}+(1+x)^2$
- What is a transformer?


---

# Related Works

- HyperDiffusion
- MeshGPT (Transformer (GPT Style Architectures (Decoder Only)))

- Neural Fields
- Meta Learning on Neural Networks (in Graph representation)

---

# Method (Storyline)


