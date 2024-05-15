"""
Training script for training regression transformers on Neural Fields
using a continous loss function

Loosely based on the nanoGPT training script

TODO:
- ddp training
"""

from contextlib import nullcontext
from dataclasses import dataclass
import os
from networks.regression_transformer import (
    RegressionTransformerConfig,
    RegressionTransformer,
)
import torch
import time
import math
import wandb
from utils import get_default_device


@dataclass
class Config:
    out_dir: str = "models"
    device: torch.device = get_default_device()
    init_from = "scratch"  # 'scratch' or 'resume'
    compile = False  # compile the model with torch.jit
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

    # training loop
    eval_interval = 2000
    log_interval = 1
    eval_iters = 200
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = True  # if True, always save a checkpoint after each eval

    wandb_project = "regression-transformer"
    wandb_run_name = "run-" + time.strftime("%Y-%m-%d-%H-%M-%S")

    # adamw optimizer
    learning_rate = 6e-4  # max learning rate
    max_iters = 600000  # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 2000  # how many steps to warm up for
    lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
    min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, config: Config):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (
        config.lr_decay_iters - config.warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def train(model_config: RegressionTransformerConfig, config: Config):

    os.makedirs(config.out_dir, exist_ok=True)
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.dtype]
    ctx = (
        torch.amp.autocast(device_type=config.device.type, dtype=ptdtype)
        if config.device.type == "cuda"
        else nullcontext()
    )

    device = config.device

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    if config.init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")

        model = RegressionTransformer(model_config)
    elif config.init_from == "resume":
        print(f"Resuming training from {config.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(config.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_config[k] = checkpoint_model_args[k]
        # create the model
        model = RegressionTransformer(model_config)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    model.to(device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        config.weight_decay,
        config.learning_rate,
        (config.beta1, config.beta2),
        config.device.type,
    )

    if config.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    # compile the model
    if config.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # let's init wandb (use both the config and the model config)
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={**vars(config), **vars(model_config)},
    )

    pass
