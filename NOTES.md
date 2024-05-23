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