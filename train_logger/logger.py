import os
import sys
import logging
from datetime import datetime
from termcolor import colored


class TrainLogger:
    """
    Custom training logger for experiments.

    Features:
    - Writes logs to both console and file.
    - Adds colors for console readability.
    - Rank-safe (only logs on rank 0 if DDP is used).
    """

    def __init__(self, log_dir="./logs", log_name=None, rank: int = 0):
        self.rank = rank
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_name = log_name or f"train_{timestamp}.log"
        self.log_path = os.path.join(log_dir, log_name)

        # Configure Python logger
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler(sys.stdout),
            ],
        )

        # Initial message
        if self.rank == 0:
            self.info(f"📝 Logging initialized — file saved at: {self.log_path}")

    # ------------------------------------------------------------------
    # Standard logging wrappers
    # ------------------------------------------------------------------
    def info(self, msg: str):
        """Prints and logs informational messages."""
        if self.rank == 0:
            print(colored(f"[INFO] {msg}", "cyan"))
            logging.info(msg)

    def warn(self, msg: str):
        """Prints and logs warnings."""
        if self.rank == 0:
            print(colored(f"[WARNING] {msg}", "yellow"))
            logging.warning(msg)

    def error(self, msg: str):
        """Prints and logs errors."""
        if self.rank == 0:
            print(colored(f"[ERROR] {msg}", "red", attrs=["bold"]))
            logging.error(msg)

    def success(self, msg: str):
        """Prints success or completion messages."""
        if self.rank == 0:
            print(colored(f"[SUCCESS] {msg}", "green"))
            logging.info(msg)

    def metric(self, epoch, train_loss, val_loss, optimizer):
        """Convenient formatted logging for training metrics."""
        if self.rank == 0:
            lrs = [group['lr'] for group in optimizer.param_groups]
            msg = f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={lrs:.6f}"
            print(colored(f"[METRIC] {msg}", "magenta"))
            logging.info(msg)

    def save_message(self, msg: str):
        """Write arbitrary message to log file."""
        if self.rank == 0:
            with open(self.log_path, "a") as f:
                f.write(f"{msg}\n")

    def get_log_path(self):
        """Return the path to the current log file."""
        return self.log_path
