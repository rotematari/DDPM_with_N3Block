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
import datetime
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
args = argparse.ArgumentParser()
args.add_argument('--checkpoints_dir', type=str, default='./checkpoints_dir')
args.add_argument('--batch_size', type=int, default=64)
args.add_argument('--suffix', type=str, default='KNN')
args.add_argument('--diffusion_steps', type=int, default=1000)
args.add_argument('--time_scale', type=float, default=1000)
args.add_argument('--img_size', type=int, default=32)
args.add_argument('--seed', type=int, default=50)
args.add_argument('--use_ema', action='store_true',default=True)
args.add_argument('--n_samples', type=int, default=64)
args.add_argument('--out_dir', type=str, default='./gen/new/')
args.add_argument('--prediction_type', type=str, default='epsilon', choices=['x0', 'epsilon', 'mu', 'x0+epsilon'])
args.add_argument('--save_separate', action='store_true')
args.add_argument('--k_neabors', type=int, default=4)
args.add_argument('--patch_size', type=list, default=[4,4,0,0])
args.add_argument('--stride', type=list, default=[2,2,0,0])
args = args.parse_args()

def main(config):
    
    args.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    UDPM = diffusion.diffusion(config=args, device=args.device)

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
            k_neabors           = config.k_neabors,
            patch_size          = config.patch_size,
            stride              = config.stride,
                            )


    # model  = MLP()

    state_dict = torch.load(os.path.join(args.checkpoints_dir, f"model{'_ema' if args.use_ema else ''}_state_{args.suffix}.pt"),
                                map_location="cpu")
    model.load_state_dict(state_dict if args.use_ema else state_dict['model'])
    model.eval()
    model.to(args.device)
    size = 0
    for p in model.parameters():
        size += p.numel()

    if args.use_ema:
            print(f"Loaded checkpoint for EMA shared model with {size / 1e6}M parameters")
    else:
        print(
        f"Loaded checkpoint for shared model  -> {size / 1e6}M parameters")

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    now = datetime.datetime.now()
    for i in tqdm(range(args.n_samples // args.batch_size)):
        x = UDPM.sample(model, batch_size=args.batch_size)
        name = str(i)
        if args.use_ema:
            name += '_ema'
        if args.save_separate:
            for b, x_ in enumerate(x):
                # x_np = np.array(x_.cpu())
                # plt.plot(x_np[0])
                # plt.savefig(os.path.join(args.out_dir, f'generated_{name}_{str(b)}.png'))
                torchvision.utils.save_image(0.5 * x_ + 0.5, args.out_dir + f'generated_{b}_{i}.png')
        else:
            # for x_ in x:
            #     plt.plot(np.arange(0, 1), np.array(0.5*x_[0].cpu() + 0.5))
            # plt.grid()
            # plt.savefig(os.path.join(args.out_dir, f'generated_{name}.png'))
            # plt.close()
            torchvision.utils.save_image(0.5*x + 0.5, args.out_dir + f"generated_{i+1}_{now.month}_{now.day}_{now.hour}:{now.minute}_{args.prediction_type}.png", normalize=True, nrows=4)
        print(f"\rGenerated_{now.month}_{now.day}_{now.hour}/{args.n_samples // args.batch_size}", end='')
        # print(f"\rGenerated {i + 1}/{args.classes_num}", end='')

    print('\nFinished')

if __name__ == "__main__":
    # args = args.parse_args()
    main(args)