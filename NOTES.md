# LLM Search

## Meetings

### 26.04 - Kickoff meeting
#### Pre-meeting notes
- Looked into state of the art papers

DiffGPT:
- Should we reimplement the two papers
- 

- Tokenization?
- Loss of training - Cross Entropy (with tokenizer) / or L2 with continuos
- Context Size?
- 
- Underfitted NeRF contain more structure

- Leading research questions? 
  - Different tokenization methods for MLP to build to vocab
  - Transfer Learning - Text model to our modality
  - Against State-of-art: Visual comparison
  
- In comparison to HyperDiffusion: More diverse shape (measure?), How far away from dataset (measure?)

- nanoGPT as baseline / Huggingface transformer implementation / Lucidrains

- Research datasets (neural fields on images?)
- GPT from hyperdiffusion
- Undercondition
- BeamSampling / Autoregressive sampling? 


### 03.05 - Meeting

- Introduction should end with our proposed method
- Contribution should be "Neo Tokenizer for Neural Fields"
- What was the problem in "Proposed Methods"
  - why do we need the proposed approaches (which problem does it fix?)
- Cut some of "Introduction" and "Proposed Methods" and "Experiments" Layout subsection in text
- "Experiments" should be more detailed- also remove unnecassary newlines
- Fastest option to get started with transformers: Huggingface

### 10.05 - Meeting

- Research Proposal?

- teacher forcing

- Ask about dataset
- Regression Transformer



## Paper

https://link.springer.com/chapter/10.1007/978-3-031-20068-7_13


## 24.05

# Main Challenges

- Tokenization of Neural Fields
- finding useful embedding that encodes the structure of the MLP 

## 27.05

### Overview

- Finish dws loading
- Regression Transformer (Overfitting on one sample) - Show predicted sequence
  -> discuss how it is not possible to fit to one sample
- Autoencoder (1dimensional, layer encoding, bias flag, positional encoding, vector quantize, residualvq (multiple tokens/codes))
- Only Codebook
- ResidualVQ with whole batch

- Weight init from hyperdiff (why?)
  - two approches:
    - using pretrained weights with the hypothesis that the trained networks follow a similar structure
    - finding permutation invariant embedding using graph neural networks

- MNIST Overfitting (Difference unconditioned/conditioned)

- Regression Transformer (multiple samples) (avg problem)

### Outlook

- Naive Descritiation of the model
- Graph (?)

## Presentation

- Visualization methods (video animation)
- Storyline (?)


### Questions

- HyperDiffusion: Why one param 8 tokens (where in code)
- Why no params in codebook (why no function on 1 dimension)
- What is a good metrix for the images? We noticed that the loss can decrease further while the reconstruction is still bad


### Meeting

- get distribution of all weights
- do both VQ-VAE and naive dicretization