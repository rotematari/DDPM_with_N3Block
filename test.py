import torch
import matplotlib.pyplot as plt
import torchvision.utils

from dataset import unlabeled_data
import argparse
import os
import random
import numpy as np
# from diffusers.models.unet_1d import UNet1DModel
from EDM_nets import DhariwalUNet
import diffusion
from tqdm import tqdm 
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
args = argparse.ArgumentParser()
args.add_argument('--checkpoints_dir', type=str, default='./checkpoints_dir')
args.add_argument('--batch_size', type=int, default=16)
args.add_argument('--suffix', type=str, default='')
args.add_argument('--diffusion_steps', type=int, default=1000)
args.add_argument('--time_scale', type=float, default=1000)
args.add_argument('--img_size', type=int, default=32)
args.add_argument('--seed', type=int, default=0)
args.add_argument('--use_ema', action='store_true',default=False)
args.add_argument('--n_samples', type=int, default=16)
args.add_argument('--out_dir', type=str, default='./gen/')
args.add_argument('--prediction_type', type=str, default='epsilon', choices=['x0', 'epsilon', 'mu', 'x0+epsilon'])
args.add_argument('--save_separate', action='store_true')
args.add_argument('--k_neabors', type=int, default=4)
args.add_argument('--patch_size', type=list, default=[4,4,0,0])
args.add_argument('--stride', type=list, default=[2,2,0,0])
args = args.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


UDPM = diffusion.diffusion(config=args, device=device)

model = DhariwalUNet(img_resolution=args.img_size,                     # Image resolution at input/output.
        in_channels=3,                        # Number of color channels at input.
        out_channels=6 if args.prediction_type == 'x0+epsilon' else 3 ,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.
        model_channels      = 64,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,
        k_neabors           = 4,
        patch_size          = [4,4,0,0],
        stride              = [2,2,0,0],
                        )



state_dict = torch.load(os.path.join(args.checkpoints_dir, f"model{'_ema' if args.use_ema else ''}_state_{args.suffix}.pt"),
                            map_location=device)
model.load_state_dict(state_dict if args.use_ema else state_dict['model'])
model.eval()
model.to(device)

xt = torch.randn(16, 3, 32, 32, dtype=torch.float32, device=device)

start = 1e-4
end = 1 - 1e-4
ts = torch.linspace(start, end, steps=1000).to(device)
delta_t = ts[1] - ts[0]
ts = ts.flip(0)
ts = torch.cat((ts, torch.zeros_like(ts[0:1])), dim=0)



for i, t in tqdm(enumerate(ts)):
    
    with torch.autocast( device_type='cuda', dtype=torch.float32):
        model_out = model(xt, 1000*t)