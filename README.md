# NeF-GPT: Autoregressive Generation of Neural Field Weights using a decoder-only Transformer

This repository contains the code for the project "NeF-GPT" for the course "Advanced Deep Learning for Computer Vision" at the Technical University of Munich.

## Environment

This project uses poetry and Python 3.10. To install the dependencies, run the following command:

```bash
poetry install
```

Refer to the poetry documentation for more information on how to install and setup the environment.

Additionally, a `requirements.txt` file is provided for compatibility with other package managers.

## Structure
The main entry points to the project are as follows:

- `generation`:
  - `train_mnist.py`: Training script to overfit implicit neural fields on MNIST brightness data
  - `train_shapenet.py`: Training script to overfit implicit neural fields on ShapeNet SDF 
    - expects point clouds to be precomputed which can be done by executing `data/pointcloud_dataset.py`. 
    - Meshes can be downloaded from Huggingface and are made watertight using the ManifoldPlus project
- `quantization`
  - Scripts used to train various quantization methods
  - `train_rq_shapenet_learnable.py`: Training script to quantize the ShapeNet NeFs using the learnable quantization method
    - Quantization schema used in the final pipeline with the correct hyperparameters
- `train_regression_transformer.ipynb`: Training script to fit the described regression transformer on the MNIST NeFs
- `train_transformer_mnist.ipynb`: Training script to fit the decoder-only Transformer on MNIST NeFs
- `train_transformer_shapenet.ipynb`: Training script to fit the decoder-only Transformer on ShapeNet NeFs

## Submissions

Under the `submissions` folder, you can find the main deliverables of the course, they have the following structure:
- `report`: LaTeX project containing the report of the project
- `proposal`: LaTeX project containing the proposal of the project
- `animation-factory`: A [motion-canvas](https://motioncanvas.io/) project containing visualizations for the presentations
- `presentation_1`: [Slidev](https://sli.dev/) slides for the first presentation
- `presentation_2`: [Slidev](https://sli.dev/) slides for the second presentation
- `poster.pdf`: The exported poster slide

Please note that the presentation slides, because of their interactive nature, the exported pdf's might not be as informative as the actual slides. Please refer to GitHub for a hosted version of the actual slides.

Hosted versions of the presentations can be found at the following links:
- Presentation 1 - TBD
- [Presentation 2](https://adl4cv-presentation-2.vercel.app)

## Models folder

This folder contains the generative models used in the project. The models are hosted on Google Drive and can be downloaded from the following link:

[models.zip](https://drive.google.com/file/d/1YOnKiq1GEGEQyidtgundzq1IQtTqXOO7/view?usp=sharing)

## Datasets folder

For script execution, the `dataset` folder is expected to contain the NeF models for the ShapeNet and MNIST dataset. The models can be built using the provided scripts or can be downloaded from the following link:

[datasets.zip](https://drive.google.com/file/d/1TS-dhVHQX-ggCM92qpaEm_zt-IqzJeka/view?usp=sharing)

## Other Folders

- `animation`: Contains the code for the animations used in the submissions
- `data`: Contains the code for the custom data sets
- `networks`: Implementation of the Transformer architectures and MLPs
- `training`: Main training scripts for the Jupyter notebooks
- `utils`: Various reusable code components used across the project