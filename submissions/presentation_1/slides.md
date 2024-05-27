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
# Introduction


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

- Goal: generative modeling of neural fields
$P(\Theta_{i} \mid \Theta_{i-1}, \Theta_{i-2}, \ldots, \Theta_{0})$



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

<VideoPane :videos="[['/n_1_type_unconditioned_model_big_idx_0.mp4'], ['/n_4_type_unconditioned_model_big_idx_0.mp4', '/n_4_type_unconditioned_model_big_idx_1.mp4', '/n_4_type_unconditioned_model_big_idx_2.mp4', '/n_4_type_unconditioned_model_big_idx_3.mp4']]" />

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
Using random initialization

<VideoPane :videos="[]" />

---
---

# Overfitting Neural Fields
Using pretrained initialization

<VideoPane :videos="[]" />

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
<VideoPane :videos="[]" />


---
---

# Challenges: Tokenization
We run into issues regarding special tokens (what comes after the start token in the absence of the start token)




---
---

# Outlok: Token Prediction with Transformer

---
---


# Regression Transformer (Unconditioned Init)

<VideoPane :videos="[['/n_1_type_pretrained_model_big_idx_0.mp4'], ['/n_4_type_pretrained_model_big_idx_0.mp4', '/n_4_type_pretrained_model_big_idx_1.mp4', '/n_4_type_pretrained_model_big_idx_2.mp4', '/n_4_type_pretrained_model_big_idx_3.mp4']]" />

---
---

# Welcome to Slidev

Presentation slides for developers

<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Press Space for next page <carbon:arrow-right class="inline"/>
  </span>
</div>

<div class="abs-br m-6 flex gap-2">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:edit />
  </button>
  <a href="https://github.com/slidevjs/slidev" target="_blank" alt="GitHub" title="Open in GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---
transition: fade-out
---

# What is Slidev?

Slidev is a slides maker and presenter designed for developers, consist of the following features

- 📝 **Text-based** - focus on the content with Markdown, and then style them later
- 🎨 **Themable** - theme can be shared and used with npm packages
- 🧑‍💻 **Developer Friendly** - code highlighting, live coding with autocompletion
- 🤹 **Interactive** - embedding Vue components to enhance your expressions
- 🎥 **Recording** - built-in recording and camera view
- 📤 **Portable** - export into PDF, PPTX, PNGs, or even a hostable SPA
- 🛠 **Hackable** - anything possible on a webpage

<br>
<br>

Read more about [Why Slidev?](https://sli.dev/guide/why)

<!--
You can have `style` tag in markdown to override the style for the current page.
Learn more: https://sli.dev/guide/syntax#embedded-styles
-->

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

<!--
Here is another comment.
-->

---
transition: slide-up
level: 2
---

# Navigation

Hover on the bottom-left corner to see the navigation's controls panel, [learn more](https://sli.dev/guide/navigation.html)

## Keyboard Shortcuts

|     |     |
| --- | --- |
| <kbd>right</kbd> / <kbd>space</kbd>| next animation or slide |
| <kbd>left</kbd>  / <kbd>shift</kbd><kbd>space</kbd> | previous animation or slide |
| <kbd>up</kbd> | previous slide |
| <kbd>down</kbd> | next slide |

<!-- https://sli.dev/guide/animations.html#click-animations -->
<img
  v-click
  class="absolute -bottom-9 -left-7 w-80 opacity-50"
  src="https://sli.dev/assets/arrow-bottom-left.svg"
  alt=""
/>
<p v-after class="absolute bottom-23 left-45 opacity-30 transform -rotate-10">Here!</p>

---
layout: two-cols
layoutClass: gap-16
---

# Table of contents

You can use the `Toc` component to generate a table of contents for your slides:

```html
<Toc minDepth="1" maxDepth="1"></Toc>
```

The title will be inferred from your slide content, or you can override it with `title` and `level` in your frontmatter.

::right::

<Toc v-click minDepth="1" maxDepth="2"></Toc>

---
layout: image-right
image: https://cover.sli.dev
---

# Code

Use code snippets and get the highlighting directly, and even types hover![^1]

```ts {all|5|7|7-8|10|all} twoslash
// TwoSlash enables TypeScript hover information
// and errors in markdown code blocks
// More at https://shiki.style/packages/twoslash

import { computed, ref } from 'vue'

const count = ref(0)
const doubled = computed(() => count.value * 2)

doubled.value = 2
```

<arrow v-click="[4, 5]" x1="350" y1="310" x2="195" y2="334" color="#953" width="2" arrowSize="1" />

<!-- This allow you to embed external code blocks -->
<<< @/snippets/external.ts#snippet

<!-- Footer -->
[^1]: [Learn More](https://sli.dev/guide/syntax.html#line-highlighting)

<!-- Inline style -->
<style>
.footnotes-sep {
  @apply mt-5 opacity-10;
}
.footnotes {
  @apply text-sm opacity-75;
}
.footnote-backref {
  display: none;
}
</style>

<!--
Notes can also sync with clicks

[click] This will be highlighted after the first click

[click] Highlighted with `count = ref(0)`

[click:3] Last click (skip two clicks)
-->

---
level: 2
---

# Shiki Magic Move

Powered by [shiki-magic-move](https://shiki-magic-move.netlify.app/), Slidev supports animations across multiple code snippets.

Add multiple code blocks and wrap them with <code>````md magic-move</code> (four backticks) to enable the magic move. For example:

````md magic-move
```ts {*|2|*}
// step 1
const author = reactive({
  name: 'John Doe',
  books: [
    'Vue 2 - Advanced Guide',
    'Vue 3 - Basic Guide',
    'Vue 4 - The Mystery'
  ]
})
```

```ts {*|1-2|3-4|3-4,8}
// step 2
export default {
  data() {
    return {
      author: {
        name: 'John Doe',
        books: [
          'Vue 2 - Advanced Guide',
          'Vue 3 - Basic Guide',
          'Vue 4 - The Mystery'
        ]
      }
    }
  }
}
```

```ts
// step 3
export default {
  data: () => ({
    author: {
      name: 'John Doe',
      books: [
        'Vue 2 - Advanced Guide',
        'Vue 3 - Basic Guide',
        'Vue 4 - The Mystery'
      ]
    }
  })
}
```

Non-code blocks are ignored.

```vue
<!-- step 4 -->
<script setup>
const author = {
  name: 'John Doe',
  books: [
    'Vue 2 - Advanced Guide',
    'Vue 3 - Basic Guide',
    'Vue 4 - The Mystery'
  ]
}
</script>
```
````

<!--
jnk
-->

---

# Components

<div grid="~ cols-2 gap-4">
<div>

You can use Vue components directly inside your slides.

We have provided a few built-in components like `<Tweet/>` and `<Youtube/>` that you can use directly. And adding your custom components is also super easy.

```html
<Counter :count="10" />
```

<!-- ./components/Counter.vue -->
<Counter :count="10" m="t-4" />

Check out [the guides](https://sli.dev/builtin/components.html) for more.

</div>
<div>

```html
<Tweet id="1390115482657726468" />
```

<Tweet id="1390115482657726468" scale="0.65" />

</div>
</div>

<!--
Presenter note with **bold**, *italic*, and ~~striked~~ text.

Also, HTML elements are valid:
<div class="flex w-full">
  <span style="flex-grow: 1;">Left content</span>
  <span>Right content</span>
</div>
-->

---
class: px-20
---

# Themes

Slidev comes with powerful theming support. Themes can provide styles, layouts, components, or even configurations for tools. Switching between themes by just **one edit** in your frontmatter:

<div grid="~ cols-2 gap-2" m="t-2">

```yaml
---
theme: default
---
```

```yaml
---
theme: seriph
---
```

<img border="rounded" src="https://github.com/slidevjs/themes/blob/main/screenshots/theme-default/01.png?raw=true" alt="">

<img border="rounded" src="https://github.com/slidevjs/themes/blob/main/screenshots/theme-seriph/01.png?raw=true" alt="">

</div>

Read more about [How to use a theme](https://sli.dev/themes/use.html) and
check out the [Awesome Themes Gallery](https://sli.dev/themes/gallery.html).

---

# Clicks Animations

You can add `v-click` to elements to add a click animation.

<div v-click>

This shows up when you click the slide:

```html
<div v-click>This shows up when you click the slide.</div>
```

</div>

<br>

<v-click>

The <span v-mark.red="3"><code>v-mark</code> directive</span>
also allows you to add
<span v-mark.circle.orange="4">inline marks</span>
, powered by [Rough Notation](https://roughnotation.com/):

```html
<span v-mark.underline.orange>inline markers</span>
```

</v-click>

<div mt-20 v-click>

[Learn More](https://sli.dev/guide/animations#click-animations)

</div>

---

# Motions

Motion animations are powered by [@vueuse/motion](https://motion.vueuse.org/), triggered by `v-motion` directive.

```html
<div
  v-motion
  :initial="{ x: -80 }"
  :enter="{ x: 0 }"
  :click-3="{ x: 80 }"
  :leave="{ x: 1000 }"
>
  Slidev
</div>
```

<div class="w-60 relative">
  <div class="relative w-40 h-40">
    <img
      v-motion
      :initial="{ x: 800, y: -100, scale: 1.5, rotate: -50 }"
      :enter="final"
      class="absolute inset-0"
      src="https://sli.dev/logo-square.png"
      alt=""
    />
    <img
      v-motion
      :initial="{ y: 500, x: -100, scale: 2 }"
      :enter="final"
      class="absolute inset-0"
      src="https://sli.dev/logo-circle.png"
      alt=""
    />
    <img
      v-motion
      :initial="{ x: 600, y: 400, scale: 2, rotate: 100 }"
      :enter="final"
      class="absolute inset-0"
      src="https://sli.dev/logo-triangle.png"
      alt=""
    />
  </div>

  <div
    class="text-5xl absolute top-14 left-40 text-[#2B90B6] -z-1"
    v-motion
    :initial="{ x: -80, opacity: 0}"
    :enter="{ x: 0, opacity: 1, transition: { delay: 2000, duration: 1000 } }">
    Slidev
  </div>
</div>

<!-- vue script setup scripts can be directly used in markdown, and will only affects current page -->
<script setup lang="ts">
const final = {
  x: 0,
  y: 0,
  rotate: 0,
  scale: 1,
  transition: {
    type: 'spring',
    damping: 10,
    stiffness: 20,
    mass: 2
  }
}
</script>

<div
  v-motion
  :initial="{ x:35, y: 30, opacity: 0}"
  :enter="{ y: 0, opacity: 1, transition: { delay: 3500 } }">

[Learn More](https://sli.dev/guide/animations.html#motion)

</div>

---

# LaTeX

LaTeX is supported out-of-box powered by [KaTeX](https://katex.org/).

<br>

Inline $\sqrt{3x-1}+(1+x)^2$

Block
$$ {1|3|all}
\begin{array}{c}

\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} &
= \frac{4\pi}{c}\vec{\mathbf{j}}    \nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\

\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} & = \vec{\mathbf{0}} \\

\nabla \cdot \vec{\mathbf{B}} & = 0

\end{array}
$$

<br>

[Learn more](https://sli.dev/guide/syntax#latex)

---

# Diagrams

You can create diagrams / graphs from textual descriptions, directly in your Markdown.

<div class="grid grid-cols-4 gap-5 pt-4 -mb-6">

```mermaid {scale: 0.5, alt: 'A simple sequence diagram'}
sequenceDiagram
    Alice->John: Hello John, how are you?
    Note over Alice,John: A typical interaction
```

```mermaid {theme: 'neutral', scale: 0.8}
graph TD
B[Text] --> C{Decision}
C -->|One| D[Result 1]
C -->|Two| E[Result 2]
```

```mermaid
mindmap
  root((mindmap))
    Origins
      Long history
      ::icon(fa fa-book)
      Popularisation
        British popular psychology author Tony Buzan
    Research
      On effectiveness<br/>and features
      On Automatic creation
        Uses
            Creative techniques
            Strategic planning
            Argument mapping
    Tools
      Pen and paper
      Mermaid
```

```plantuml {scale: 0.7}
@startuml

package "Some Group" {
  HTTP - [First Component]
  [Another Component]
}

node "Other Groups" {
  FTP - [Second Component]
  [First Component] --> FTP
}

cloud {
  [Example 1]
}

database "MySql" {
  folder "This is my folder" {
    [Folder 3]
  }
  frame "Foo" {
    [Frame 4]
  }
}

[Another Component] --> [Example 1]
[Example 1] --> [Folder 3]
[Folder 3] --> [Frame 4]

@enduml
```

</div>

[Learn More](https://sli.dev/guide/syntax.html#diagrams)

---
foo: bar
dragPos:
  square: 691,33,167,_,-16
---

# Draggable Elements

Double-click on the draggable elements to edit their positions.

<br>

###### Directive Usage

```md
<img v-drag="'square'" src="https://sli.dev/logo.png">
```

<br>

###### Component Usage

```md
<v-drag text-3xl>
  <carbon:arrow-up />
  Use the `v-drag` component to have a draggable container!
</v-drag>
```

<v-drag pos="671,205,253,_,-15">
  <div text-center text-3xl border border-main rounded>
    Double-click me!
  </div>
</v-drag>

<img v-drag="'square'" src="https://sli.dev/logo.png">

---
src: ./pages/multiple-entries.md
hide: false
---

---

# Monaco Editor

Slidev provides built-in Monaco Editor support.

Add `{monaco}` to the code block to turn it into an editor:

```ts {monaco}
import { ref } from 'vue'
import { emptyArray } from './external'

const arr = ref(emptyArray(10))
```

Use `{monaco-run}` to create an editor that can execute the code directly in the slide:

```ts {monaco-run}
import { version } from 'vue'
import { emptyArray, sayHello } from './external'

sayHello()
console.log(`vue ${version}`)
console.log(emptyArray<number>(10).reduce(fib => [...fib, fib.at(-1)! + fib.at(-2)!], [1, 1]))
```

---
layout: center
class: text-center
---

# Learn More

[Documentations](https://sli.dev) · [GitHub](https://github.com/slidevjs/slidev) · [Showcases](https://sli.dev/showcases.html)

