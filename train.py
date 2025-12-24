from clr_utils import *
import json
from torch.utils.data import DataLoader, DistributedSampler
from ddp.env import setup_env, seed_everything
from ddp.parse import parse_ddp_args, init_distributed_mode
from utils.data_aug_config import ThermalAugConfig, RgbAugConfig
from utils.dataloader import (
    get_datasets_and_loaders,
    ThermalAugmentation,
    RgbAugmentation,
)
from ddp.dist_util import (
    is_dist,
    get_world_size,
    get_rank,
    wrap_model,
    get_model_device,
)
from train_logger.logger import TrainLogger
from model.vit import VisionTransformer
from model.load_weight import load_pretrained_vit_weights
from utils.scheduler import get_optimizer, cosine_schedule
from spatialcl.uwcl import build_uwcl
from check_point.save_point import save_checkpoint


def param_dataloader_init(args, logger: TrainLogger = None):
    """Main training setup for distributed or single-node training."""

    # --- Select transformation based on modality ---
    if isinstance(args.modality, str):
        args.modality = json.loads(args.modality)
    else:
        logger.error("Modality argument must be a string in JSON format.")
    if args.modality.get("rgb", False):
        transform = RgbAugmentation(RgbAugConfig()).transform
    elif args.modality.get("thermal", False):
        transform = ThermalAugmentation(ThermalAugConfig()).transform
    else:
        raise ValueError(
            "Please specify at least one valid modality: 'rgb' or 'thermal'."
        )

    # --- Get Datasets ---

    train_dataset, val_dataset, test_dataset = get_datasets_and_loaders(
        root=args.root,
        dataset_class= None if args.dataset_class is [None, "None"] else eval(args.dataset_class),
        transform=transform,
    )

    # --- Samplers (DDP-aware) ---
    def build_sampler(dataset):
        if is_dist():
            return DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
                drop_last=True,
            )
        return None

    sampler_train = build_sampler(train_dataset)
    sampler_val = build_sampler(val_dataset)
    sampler_test = build_sampler(test_dataset)

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler_train is None),
        sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers = True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler_val,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers = True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler_test,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers = True
    )
    return train_loader, val_loader, test_loader


def one_epoch_train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: TrainLogger,
    args,
):
    """One epoch training loop."""
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(train_loader):
        (x1, x2), labels, img_ids = batch
        x1, x2, labels, img_ids = (
            x1.to(device),
            x2.to(device),
            labels.to(device),
            img_ids.to(device),
        )

        images = torch.cat([x1, x2], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        img_ids = torch.cat([img_ids, img_ids], dim=0)

        optimizer.zero_grad()
        z = model(images)

        loss = build_uwcl(
            z=z,
            labels=labels,
            epoch=epoch,
            img_ids=img_ids,
            device=device,
            temperature=args.temperature,
            T=args.num_epochs,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x1.size(0)

        if step % 20 == 0:
            logger.info(
                f"Epoch [{epoch}] Step [{step}/{len(train_loader)}]: Loss = {loss.item():.4f}"
            )

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss


# =====================
def one_eval_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    logger: TrainLogger,
    args,
):
    """One epoch evaluation loop."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            (x1, x2), labels, img_ids = batch
            x1, x2, labels, img_ids = (
                x1.to(device),
                x2.to(device),
                labels.to(device),
                img_ids.to(device),
            )

            images = torch.cat([x1, x2], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            img_ids = torch.cat([img_ids, img_ids], dim=0)

            z = model(images)
            loss = loss = build_uwcl(
                z=z,
                epoch=epoch,
                labels=labels,
                img_ids=img_ids,
                device=device,
                temperature=args.temperature,
                T=args.num_epochs,
            )
            total_loss += loss.item() * x1.size(0)
            if step % 20 == 0:
                logger.info(
                    f"[Eval] Epoch [{epoch}] Step [{step}/{len(val_loader)}]: Loss = {loss.item():.4f}"
                )

    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss


# =====================


def main():
    # --- Environment setup ---
    setup_env()
    seed_everything()
    
    rank = get_rank() if torch.distributed.is_initialized() else 0
    logger = TrainLogger(log_dir="./logs", rank=rank)
    logger.info("Environment setup complete.")
    # --- Distributed setup ---
    args = parse_ddp_args()
    if args.is_distributed:
        init_distributed_mode(args,logger)

    """Main training setup for distributed or single-node training."""
    # --- DataLoaders & DDP setup ---
    train_loader, val_loader, _ = param_dataloader_init(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Starting training...")
    # --- Model, criterion, optimizer ---
    if is_dist():
        logger.info(
            f"Distributed training initialized. World Size: {get_world_size()}, Rank: {get_rank()}"
        )
        model = wrap_model(VisionTransformer(variant=args.vit_variant))
    else:
        model = VisionTransformer(variant=args.vit_variant).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("Normal training, Distributed training not initialized.")

    weighted_model = load_pretrained_vit_weights(
        custom_model=model, model_size=args.vit_variant, device=device
    )
    optimizer = get_optimizer(model=weighted_model)

    best_val_loss = float("inf")

    # --- Training loop ---
    logger.info("Beginning training loop...")
    for epoch in range(args.num_epochs):
        cosine_schedule(
            epoch=epoch,
            optimizer=optimizer,
            max_epochs=args.num_epochs,
            warmup_epochs=args.warmup_epochs,
        )

        train_loss = one_epoch_train(
            model, train_loader, optimizer, device, epoch, logger, args
        )
        val_loss = one_eval_epoch(model, val_loader, device, epoch, logger, args)
        logger.metric(epoch, train_loss, val_loss, optimizer)

        if val_loss < best_val_loss:
            logger.success("new weight saved!")
            best_val_loss = val_loss
            is_best = True
            save_checkpoint(
                state={
                    "model_state_dict": model.state_dict(),
                },
                is_best=is_best,
                checkpoint_dir="./checkpoints",
                filename="best_last.pth",
            )
        else:
            logger.info("No improvement this epoch.")


if __name__ == "__main__":
    main()
