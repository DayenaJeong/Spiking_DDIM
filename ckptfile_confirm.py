import os
import hashlib
import torch
import torch.nn as nn
import argparse
import sys

sys.path.append('/')
from models.diffusion import Model  # Assuming you have a Model class defined in models.diffusion
from models.ema import EMAHelper  # Assuming you have an EMAHelper class defined


def md5_hash(path):
    """Calculate MD5 hash of a file to check its integrity."""
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def check_ckpt_exists(ckpt_path):
    """Check if the checkpoint file exists at the specified path."""
    if os.path.exists(ckpt_path):
        print(f"Checkpoint file found: {ckpt_path}")
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")


def print_checkpoint_structure(checkpoint):
    """Print the keys in the loaded checkpoint to verify its structure."""
    print("Checkpoint contains the following keys:")
    for key in checkpoint.keys():
        print(key)


def sample(args, config):
    """Sample function to load model and verify checkpoint."""
    print("Starting sample function...")  # È®ÀÎ¿ë ¸Þ½ÃÁö
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass the device argument when creating the model
    model = Model(config['model'], device=device)

    if not args.use_pretrained:
        # Use custom trained model's checkpoint
        ckpt_path = os.path.join(args.log_path, "ckpt.pth")
        if config.get('sampling', {}).get('ckpt_id') is not None:
            ckpt_path = os.path.join(args.log_path, f"ckpt_{config['sampling']['ckpt_id']}.pth")

        # Check if checkpoint exists
        check_ckpt_exists(ckpt_path)

        # Load the checkpoint
        try:
            states = torch.load(ckpt_path, map_location=device)
            print(f"Checkpoint loaded from {ckpt_path}")
            print(f"MD5 hash of checkpoint: {md5_hash(ckpt_path)}")
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {ckpt_path}") from e

        # Print the structure of the checkpoint
        print_checkpoint_structure(states)

        # Load the model's weights from checkpoint
        model = model.to(device)
        model = nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        print("Model state loaded successfully.")

        # Verify the first layer's weights
        print(f"First layer weights: {list(model.parameters())[0]}")

        # Load EMA helper if necessary
        if config['model'].get('ema', False):
            ema_helper = EMAHelper(mu=config['model']['ema_rate'])
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None
    else:
        # Use pretrained model
        if config['data']['dataset'] == "CIFAR10":
            ckpt = "/srv1/dayena7/cifar10_checkpoint/model-790000.ckpt"
        elif config['data']['dataset'] == "LSUN":
            ckpt = "/srv1/dayena7/lsun_bedroom_checkpoint/model-2388000.ckpt"
        elif config['data']['dataset'] == "CELEBA":
            ckpt = "/srv1/dayena7/celeba_checkpoint/celeba_600.pt"
        else:
            raise ValueError("Unsupported dataset")

        # Check if checkpoint exists
        check_ckpt_exists(ckpt)

        # Load the checkpoint
        try:
            checkpoint = torch.load(ckpt, map_location=device)
            print(f"Checkpoint loaded from {ckpt}")
            print(f"MD5 hash of checkpoint: {md5_hash(ckpt)}")
        except Exception as e:
            raise RuntimeError(f"Error loading pretrained checkpoint: {ckpt}") from e

        # Print the structure of the checkpoint
        print_checkpoint_structure(checkpoint)

        # Load the model's weights from checkpoint
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError("Checkpoint format is not supported")

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model = nn.DataParallel(model)
        print("Pretrained model loaded successfully.")

    model.eval()

    # Verify model size
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {model_size}")

    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    size = (param_size + buffer_size) / 1024 ** 2
    print(f"Model size: {size:.3f} MB")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Checkpoint verification")
    parser.add_argument("--config", required=True, help="Path to config file (YAML)")
    parser.add_argument("--log_path", required=True, help="Log path where checkpoints are stored")
    parser.add_argument("--use_pretrained", type=bool, default=False, help="Whether to use pretrained model")
    args = parser.parse_args()

    # Load your config (as a placeholder, you will need to load the actual YAML config)
    config = {
        "model": {
            "ch": 128,
            "out_ch": 3,
            "ch_mult": [1, 2, 2, 2],
            "num_res_blocks": 2,
            "attn_resolutions": [16],
            "dropout": 0.1,
            "ema": True,
            "ema_rate": 0.9999,
            "type": "simple"
        },
        "data": {
            "dataset": "CIFAR10",
            "image_size": 32,
            "channels": 3
        },
        "sampling": {}
    }

    # Call the sample function
    sample(args, config)