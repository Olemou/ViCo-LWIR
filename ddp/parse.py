import os
import argparse
import torch.distributed as dist
import torch
from torchvision import datasets
from clr_utils import *


def parse_ddp_args():
    """Parse and infer DDP args from torchrun or manual input."""
    parser = argparse.ArgumentParser(description="Distributed Training Arguments")

    # Optional manual override (torchrun automatically sets envs)
    parser.add_argument(
        "--nnodes", type=int, default=int(os.environ.get("NUMBER_NODE", 1))
    )
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="index of GPU used",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        help="Address of the master node (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=int(os.environ.get("MASTER_PORT", 29500)),
        help="Port used by the master node (default: 29500)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", 32))
    )
    parser.add_argument(
        "--num_workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4))
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--vit_variant", type=str, default="base")
    parser.add_argument(
        "--modality", type=dict, default={"rgb": True, "thermal": False}
    )
    parser.add_argument(
    "--is_distributed",
    action="store_true",
    help="Enable DistributedDataParallel training",
)

    parser.add_argument("--dataset_class",  default=datasets.CIFAR10)

    args = parser.parse_args()

    return args


def init_distributed_mode(args):
    """Initialize torch.distributed if needed."""

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for distributed training.")
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        return args

    torch.cuda.set_device(args.gpu)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        world_size=args.nnodes * args.nproc_per_node,
        rank=args.node_rank * args.nproc_per_node + args.local_rank,
    )
    dist.barrier()
    return args
