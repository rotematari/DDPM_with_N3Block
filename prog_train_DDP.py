import torch
import torchvision.datasets.folder
from torch.cuda import amp
from torch.utils.data import DataLoader
import argparse
import os
import utils
from dataset import unlabeled_data
import time
import diffusion
from EDM_nets import DhariwalUNet

import torch.distributed as dist
import torchvision.transforms as transforms
import copy
import sample_prog
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

args = argparse.ArgumentParser()
args.add_argument('--checkpoints_dir', type=str, default='./checkpoints_dir', required=False)
args.add_argument('--data_dir', type=str, default='./data_dir', required=False)
args.add_argument('--log_dir', type=str, default='./log_dir', required=False)
args.add_argument('--lr', type=float, default=5e-4)
args.add_argument('--EMA_weight', type=float, default=0.9999)
args.add_argument('--time_scale', type=float, default=1000)
args.add_argument('--batch_per_GPU', type=int, default=64)
args.add_argument('--mini_batch_size', type=int, default=None)
args.add_argument('--num_workers', type=int, default=0)
args.add_argument('--epochs', type=int, default=1500)
args.add_argument('--save_every', type=int, default=10000)
args.add_argument('--diffusion_steps', type=int, default=1000)
args.add_argument('--suffix', type=str, default='KNN')
args.add_argument('--img_size', type=int, default=32)
args.add_argument('--seed', type=int, default=0)
args.add_argument('--sync_every', type=int, default=1)
args.add_argument('--local_rank', type=int, default=0)
args.add_argument('--training_steps', type=int, default=2000000)
args.add_argument('--print_interval', type=int, default=100)
args.add_argument('--use_amp', action='store_true')
args.add_argument('--ddp_backend', type=str, default='nccl', choices=['nccl', 'mpi'])
args.add_argument('--prediction_type', type=str, default='epsilon', choices=['x0', 'epsilon', 'mu', 'x0+epsilon'])
args.add_argument('--k_neabors', type=int, default=4)
args.add_argument('--patch_size', type=list, default=[4,4,0,0])
args.add_argument('--stride', type=list, default=[2,2,0,0])
args = args.parse_args()


if args.mini_batch_size is None:
    args.mini_batch_size = args.batch_per_GPU
else:
    assert args.mini_batch_size <= args.batch_per_GPU,""
    assert args.batch_per_GPU % args.mini_batch_size == 0


args.world_size = 0

def train(args):

    gpu = 0
    torch.cuda.set_device(gpu)
    device_gpu = torch.device(gpu)
    print(f"=> set cuda device = {gpu}")
    print(device_gpu)


    print(args)
    logging.info("Command-line arguments: %s", args)

    # dataset = unlabeled_data(data_dir=args.data_dir, image_size=args.img_size)
    # print(f"Found {len(dataset)} training images.")
    TRANSFORM = transforms.Compose(
    (transforms.ToTensor(), transforms.RandomHorizontalFlip(),
    #  transforms.Normalize([0.5], [0.5])
     ))


    dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                            download=True, transform=TRANSFORM)



    dataloader = DataLoader(dataset, batch_size=args.mini_batch_size,
                            pin_memory=True, shuffle= True,
                            drop_last=True,)




    diff = diffusion.diffusion(config=args, device=device_gpu)

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
        k_neabors           = args.k_neabors,
        patch_size          = args.patch_size,
        stride              = args.stride,
                         )
   

    num_pars = 0
    for p in model.parameters():
        num_pars += p.numel()
    print(f'Created model with {num_pars/1e6: .2f}M parameters')
    logging.info(f'Created model with {num_pars/1e6: .2f}M parameters')
    model = model.train().to(device_gpu)
    logging.info("Network Configuration: %s", model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)


    model = torch.nn.DataParallel(model,device_ids=[gpu])

    if not os.path.isdir(args.checkpoints_dir) :
        os.mkdir(args.checkpoints_dir)

    start_epoch = 0
    training_steps = 0
    if os.path.isfile(os.path.join(args.checkpoints_dir, f"model_state"
                                                         f"_{args.suffix}.pt")):
        state_dict = torch.load(os.path.join(args.checkpoints_dir, f"model_"
                                                                   f"state_{args.suffix}.pt"),
                                map_location="cpu")
        model.module.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        if optimizer.param_groups[0]["lr"] != args.lr:
            for g in optimizer.param_groups:
                g['lr'] = args.lr
        start_epoch = state_dict['epochs']
        training_steps = state_dict['optimizer']['state'][1]['step'] + 1
        print(f"Loaded existing checkpoint model_state_"
              f"_{args.suffix}.pt, training steps = {training_steps}")
        if args.EMA_weight > 0 :
            ema_state_dict = torch.load(os.path.join(args.checkpoints_dir,
                                                     f"model_ema_state"
                                                     f"_{args.suffix}.pt"))
    elif args.EMA_weight > 0:
        ema_state_dict = copy.deepcopy(model.module.state_dict())

    gScaler = amp.GradScaler()
    avg_loss = 0
    optimizer.zero_grad(set_to_none=True)
    t = torch.zeros(args.mini_batch_size, dtype=torch.float32, device=device_gpu)
    step_every = args.batch_per_GPU // args.mini_batch_size
    tik = time.time()
    for epoch in range(start_epoch, args.epochs):
        for idx, data in enumerate(dataloader):
            with torch.no_grad():
                x0 = data[0].to(device_gpu, non_blocking=True)
                t.uniform_(0, 1)
                # t.random_(0, args.diffusion_steps)

                x0 = 2 * x0 - 1

                xt, eps = diff.get_xt(x0=x0, t=t.reshape(-1, 1, 1, 1))


            if (idx + 1) % step_every == 0:
                with torch.autocast(enabled=args.use_amp, device_type='cuda', dtype=torch.bfloat16 if args.use_amp else torch.float32):
                    model_out = model(xt, args.time_scale*t)
                    loss = diff.train_loss(model_out=model_out, x0=x0, eps=eps)
                gScaler.scale(loss).backward()
                gScaler.step(optimizer)
                gScaler.update()
                optimizer.zero_grad(set_to_none=True)
                training_steps += 1
            else:
                
                with torch.autocast(enabled=args.use_amp, device_type='cuda', dtype=torch.bfloat16 if args.use_amp else torch.float32):
                    model_out = model(xt, args.time_scale*t)
                    loss = diff.train_loss(model_out=model_out, x0=x0, eps=eps)
                gScaler.scale(loss).backward()

            # if gpu == 0:
            with torch.no_grad():
                avg_loss += loss.detach()
            if (training_steps+1) % min(args.print_interval, len(dataloader)) == 0 and (idx+1) % step_every == 0:
                tok = time.time()
                print(
                    f'GPU: {gpu} | '
                    f'Epoch = {epoch}/{args.epochs} | iterations = {training_steps} | '
                    f'avg loss = {avg_loss.item()/min(args.print_interval, len(dataloader)): .5f} ' 
                    f'| time = {tok - tik: .3f} '
                    f'| lr = {optimizer.param_groups[0]["lr"]}')
                avg_loss = 0
                tik = time.time()

            if args.EMA_weight > 0:
                ema_state_dict = utils.EMA_update(ema_state_dict, model.module.state_dict(), args.EMA_weight)

            if (training_steps + 1) % args.save_every == 0 :
                
                state_dict = {
                    'optimizer': optimizer.state_dict(),
                    'model': model.module.state_dict(),
                    'epochs': epoch,
                }
                torch.save(state_dict,
                           os.path.join(args.checkpoints_dir, f"{'model'}_"
                                                              f"state_{args.suffix}.pt"))
                if args.EMA_weight > 0:
                    torch.save(ema_state_dict,
                           os.path.join(args.checkpoints_dir, f"{'model'}_"
                                                              f"ema_state_{args.suffix}.pt"))
                sample_prog.main(args)

if __name__ == '__main__':
    logging.basicConfig(filename=os.path.join(args.log_dir, 'training.log'), level=logging.INFO)
    train(args)
