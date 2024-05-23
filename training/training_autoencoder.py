from contextlib import nullcontext
from dataclasses import asdict, dataclass
import os
import itertools
from networks.regression_transformer import (
    RegressionTransformerConfig,
    RegressionTransformer,
)
import torch
import time
import math
import wandb
from utils import get_default_device
from networks.naive_rq_ae import RQAutoencoder, RQAutoencoderConfig
from torch import autocast, optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


@dataclass
class TrainingConfig:
    out_dir: str = "models"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_from = "scratch"  # 'scratch' or 'resume'
    compile = False  # compile the model with torch.jit
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

    # training loop
    eval_interval = 100
    log_interval = 1
    eval_iters = 100  # 200
    eval_only = False  # if True, script exits right after the first eval
    checkpoint_interval = 100 # how often to save a checkpoint
    always_save_checkpoint = False  # if True, always save a checkpoint after each eval

    gradient_accumulation_steps = 1  # 5 * 8  # used to simulate larger batch sizes
    batch_size = 1  # 8  # effective batch size

    wandb_project = "autoencoder"
    wandb_run_name = "run-" + time.strftime("%Y-%m-%d-%H-%M-%S")

    # adamw optimizer
    learning_rate = 6e-4  # max learning rate
    max_iters = 30  # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 500  # how many steps to warm up for
    lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
    min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

def get_lr_scheduler(optimizer, config):
    def lr_lambda(current_step: int):
        if current_step < config.warmup_iters:
            return float(current_step) / float(max(1, config.warmup_iters))
        progress = float(current_step - config.warmup_iters) / float(max(1, config.lr_decay_iters - config.warmup_iters))
        return max(config.min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def train_model(
    config: TrainingConfig,
    model_config: RQAutoencoderConfig,
    train_loader,
    eval_loader,
    loss: torch.nn.Module,
):
    # Initialize Weights and Biases
    super_config = asdict(config) | asdict(model_config)
    wandb.init(
        project=config.wandb_project, name=config.wandb_run_name, config=super_config
    )

    # Set device
    device = config.device

    model = RQAutoencoder(model_config)

    # Initialize model, loss function, and optimizer
    model = model.to(device)
    criterion = loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    scaler = None
    scheduler = get_lr_scheduler(optimizer, config)

    # Training loop
    model.train()
    for epoch in range(config.max_iters):
        loop = tqdm(train_loader, leave=False)
        for i, batch in enumerate(loop):
            inputs = batch[0].to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=bool(scaler)):
                outputs = model(inputs)
                loss = criterion(outputs, inputs)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Update the learning rate
            if config.decay_lr:
                scheduler.step()

            loop.set_description(f"Epoch [{epoch + 1}/{config.max_iters}]")
            loop.set_postfix(loss=loss.item())

            # Log and evaluate
            if (i + 1) % config.log_interval == 0:
                wandb.log({"loss": loss.item(), "epoch": epoch, "batch": i + 1, "lr": scheduler.get_last_lr()[0]})


            if (i + 1) % config.eval_interval == 0:
                eval_loss = evaluate_model(
                    model, eval_loader, criterion, device, scaler
                )
                wandb.log({"eval_loss": eval_loss, "epoch": epoch, "batch": i + 1})

            if (i + 1) % config.checkpoint_interval == 0 and config.always_save_checkpoint:
                save_checkpoint(model, optimizer, epoch, config.out_dir)

    wandb.finish()


def evaluate_model(model, eval_loader, criterion, device, scaler):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in itertools.islice(eval_loader, 0, 16):
            inputs = batch[0].to(device)
            with autocast(device_type=device.type, enabled=bool(scaler)):
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                eval_loss += loss.item()

    eval_loss /= len(eval_loader)
    model.train()
    return eval_loss


def save_checkpoint(model, optimizer, epoch, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    checkpoint_path = os.path.join(out_dir, f"model_epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved: {checkpoint_path}")
