"""
Training script for training regression transformers on Neural Fields
using a continous loss function

Loosely based on the nanoGPT training script

TODO:
- ddp training
"""

from contextlib import nullcontext
from data.neural_field_datasets_shapenet import ShapeNetDataset
from training.mnist_classifier_score import compute_mnist_score
from vector_quantize_pytorch import VectorQuantize
from dataclasses import asdict, dataclass
import os
from networks.nano_gpt import (
    GPT,
    GPTConfig,
)
import torch
import time
import math
import wandb
from utils import get_default_device
import time

wandb.login()


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


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
    eval_interval = 100
    metric_interval = 100
    log_interval = 1
    eval_iters = 16  # 200
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = False  # if True, always save a checkpoint after each eval

    gradient_accumulation_steps = 1  # 5 * 8  # used to simulate larger batch sizes
    batch_size = 8  # 8  # effective batch size

    wandb_project = "naive_token_transformer"
    wandb_run_name = "run-" + time.strftime("%Y-%m-%d-%H-%M-%S")

    # adamw optimizer
    learning_rate = 6e-4  # max learning rate
    max_iters = 600000  # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 500  # how many steps to warm up for
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


def train(
    get_batch: callable,
    config: Config,
    model_config: GPTConfig,
    vq: VectorQuantize = None,
    vq_config: dict = None,
    early_stop: EarlyStopper = None,
    token_dict: dict = None,
    custom_eval: callable = lambda logits, split, k, x, y: None,
    important_sampling: bool = True,
):
    if important_sampling:
        losses_over_dataset = torch.cat([100]*len(ShapeNetDataset(os.path.join("./", "datasets", "shapenet_nefs", "pretrained"))))

        
        
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

        model = GPT(model_config)
    elif config.init_from == "resume":
        print(f"Resuming training from {config.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(config.out_dir, "funktioniert_okeisch.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        # for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        #    model_config[k] = checkpoint_model_args[k]
        # create the model
        model = GPT(checkpoint_model_args)
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

    # compute mnist metric score
    @torch.no_grad()
    def compute_metrics():
        pass # return compute_mnist_score(model, vq, device, token_dict)

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(config.eval_iters)
            for k in range(config.eval_iters):
                
                with ctx:
                    if important_sampling:
                        X, Y, idx, dataset_indices = get_batch(split, losses=losses_over_dataset)
                        logits, loss, loss_per_sample = model(X, Y, idx, reduction="none")
                        losses_over_dataset[dataset_indices] = loss_per_sample
                    else:
                        X, Y, idx = get_batch(split)
                        logits, loss = model(X, Y, idx)
                    custom_eval(logits, split, k)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # let's init wandb (use both the config and the model config)
    # merge the two dataclasses into a single dict (with defaults from the model config)
    super_config = asdict(config) | asdict(model_config)
    wandb.init(
        project=config.wandb_project, name=config.wandb_run_name, config=super_config
    )

    # training loop
    if important_sampling:
        X, Y, idx, dataset_indices = get_batch("train", losses=losses_over_dataset)
    else:
        X, Y, idx = get_batch("train")
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model  # unwrap DDP container if needed
    running_mfu = -1.0
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, config) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # # evaluate the loss on train/val sets and write checkpoints
        # if iter_num % config.metric_interval == 0:
        #     acc, metrics = compute_metrics()
        #     print(
        #         f"step {iter_num}: mnist classifier loss: {metrics}, mnist accuracy: {acc}"
        #     )
        #     wandb.log({"iter": iter_num, "mnist_loss": metrics, "mnist_acc": acc})

        if iter_num % config.eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
            )
            if losses["val"] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": model_config,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                        "vq_state_dict": vq.state_dict() if vq else {},
                        "vq_config": vq_config if vq_config else {},
                        "token_dict": token_dict if token_dict else {},
                    }
                    print(f"saving checkpoint to {config.out_dir}")
                    torch.save(checkpoint, os.path.join(config.out_dir, "ckpt.pt"))

            if early_stop:
                if early_stop(losses["val"]):
                    print("Early stopping due to increasing validation loss!")
                    return model

        if iter_num == 0 and config.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(config.gradient_accumulation_steps):
            with ctx:
                if losses_over_dataset:
                        X, Y, idx, dataset_indices = get_batch("train", losses=losses_over_dataset)
                        logits, loss, loss_per_sample = model(X, Y, idx, reduction="none")
                        losses_over_dataset[dataset_indices] = loss_per_sample
                else:
                    X, Y, idx = get_batch("train")
                    logits, loss = model(X, Y, idx)
                    
                loss = (
                    loss / config.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation

            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            if important_sampling:
                X, Y, idx, dataset_indices = get_batch("train", losses=losses_over_dataset)
            else:
                X, Y, idx = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        # if iter_num % config.log_interval == 0:
        #     # get loss as float. note: this is a CPU-GPU sync point
        #     # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        #     lossf = loss.item() * config.gradient_accumulation_steps
        #     if local_iter_num >= 5:  # let the training loop settle a bit
        #         mfu = raw_model.estimate_mfu(
        #             config.batch_size * config.gradient_accumulation_steps, dt
        #         )
        #         running_mfu = (
        #             mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        #         )
        #     print(
        #         f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        #     )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > config.max_iters:
            return model

    pass
