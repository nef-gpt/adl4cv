from animation.visualization_regression_transformer import visualize_learning_process
from training import training_regression_transformer
from networks.regression_transformer import (
    RegressionTransformerConfig,
    RegressionTransformer,
)

from data.nef_mnist_dataset import (
    DWSNetsDataset,
    FlattenMinMaxTransform,
    MnistNeFDataset,
    FlattenTransform,
    MinMaxTransform,
)

import os
import torch
import torchinfo

# Config for this script

dir_path = os.path.dirname(os.path.abspath(os.getcwd()))
data_root_ours = os.path.join(dir_path, "adl4cv", "datasets", "mnist-nerfs")
dataset_kwargs = {
    "fixed_label": 5,
}

# Config Training
training_config = training_regression_transformer.Config()
training_config.learning_rate = 5e-4
training_config.max_iters = 14000
training_config.weight_decay = 0
training_config.decay_lr = True
training_config.lr_decay_iters = 14000
training_config.warmup_iters = 0.1 * training_config.max_iters
training_config.batch_size = 1
# TODO: config.detailed_folder

big_model_config = {
    "n_embd": 32,
    "n_head": 8,
    "n_layer": 16,
}

small_model_config = {
    "n_embd": 16,
    "n_head": 4,
    "n_layer": 8,
}

types = ["unconditioned", "pretrained"]
n_values = [1, 4, 8, 16, 32]

skip_list = [
    # (big_label, n, type, idx)
    ("big", 1, "unconditioned", 0),
    ("big", 4, "unconditioned", 0),
    ("big", 4, "unconditioned", 1),
    ("big", 4, "unconditioned", 2),
]

### SCRIPT


def get_batch(split: str, n: int, block_size: int, dataset):
    # let's get a batch with the single element
    # y should be the same shifted by 1
    ix = torch.zeros(training_config.batch_size, dtype=torch.int)
    # torch.randint(torch.numel(flattened) - model_config.block_size, (config.batch_size,))

    # randomly select a sample (0...n-1)
    split_start = 0 if split == "train" else int(n)
    split_end = int(n) if split == "train" else n + training_config.batch_size

    sample = dataset[torch.randint(split_start, split_end, (1,))][0]

    x = torch.stack([sample[i : i + block_size] for i in ix])
    y = torch.stack([sample[i + 1 : i + 1 + block_size] for i in ix])

    # x and y have to be (1, *, 1)
    x = x.unsqueeze(-1).to(training_config.device)
    y = y.unsqueeze(-1).to(training_config.device)
    return x, y


def do_type(type: str):

    kwargs = {**dataset_kwargs, "type": type}

    dataset_wo_min_max = MnistNeFDataset(
        data_root_ours, transform=FlattenTransform(), **kwargs
    )
    min_ours, max_ours = dataset_wo_min_max.min_max()
    dataset = MnistNeFDataset(
        data_root_ours, transform=FlattenMinMaxTransform((min_ours, max_ours)), **kwargs
    )
    dataset_no_transform = MnistNeFDataset(data_root_ours, **kwargs)

    total_n = len(dataset)
    print(f"Dataset loaded with {len(dataset)} samples")

    # Config Transformer
    for config in [big_model_config, small_model_config]:
        big_label = "big" if config == big_model_config else "small"
        model_config = RegressionTransformerConfig(
            block_size=len(dataset[0][0]) - 1, **config
        )

        for n in n_values:

            detailed_folder = f"n_{n}_type_{type}_model_{big_label}"
            # check if detailed folder under models already exists
            if os.path.exists(f"./models/{detailed_folder}"):
                print(f"Skipping {detailed_folder} as it already exists")
            else:
                assert n < total_n, f"n={n} is larger than the dataset size {total_n}"
                print(
                    f"Training with n={n} type={type} and model_config={model_config}"
                )
                get_model_batch = lambda split: get_batch(
                    split, n, model_config.block_size, dataset
                )
                training_config.detailed_folder = detailed_folder
                training_config.wandb_run_name = f"n_{n}_type_{type}_model_{big_label}"
                training_regression_transformer.train(
                    get_model_batch, training_config, model_config
                )

            # Do the animation directly
            try:
                for idx in range(0, n):
                    if (big_label, n, type, idx) in skip_list:
                        continue
                    if idx > 3:
                        break
                    video_name = f"n_{n}_type_{type}_model_{big_label}_idx_{idx}"
                    visualize_learning_process(
                        idx,
                        training_config.max_iters,
                        detailed_folder,
                        video_name,
                        dataset_kwargs=kwargs,
                        regression_config=model_config,
                    )
            # catch all errors
            except Exception as e:
                # print with complete error info (stack trace)
                print(e)
                continue


def main():
    print("running on device {}".format(training_config.device))
    do_type("pretrained")
    # for type in types:
    #   do_type(type)


if __name__ == "__main__":
    main()
