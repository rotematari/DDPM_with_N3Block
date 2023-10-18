import torch
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
class diffusion:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        # scale = 1000 / config.diffusion_steps
        # beta_start = scale * 0.0001
        # beta_end = scale * 0.02
        # self.betas = torch.linspace(beta_start, beta_end, config.diffusion_steps, dtype=torch.float64, device=device)
        # self.alphas = 1 - self.betas
        # self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        #
        # self.ts = torch.arange(0, config.diffusion_steps, device=device, dtype=torch.long).flip(0)
        #
        # self.delta_t = 1


        if 'diffusion_steps' in config:
            start = 1e-4
            end = 1 - 1e-4
            ts = torch.linspace(start, end, steps=self.config.diffusion_steps).to(self.device)
            self.delta_t = ts[1] - ts[0]
            self.ts = ts.flip(0)
            self.ts = torch.cat((self.ts, torch.zeros_like(ts[0:1])), dim=0)

    def alpha_bar(self, t):
        # assert torch.all(0 <= t) and torch.all(t <= 1)
        # return 1 - t
        # return self.alpha_cumprod[t].float()
        return torch.exp(-t**2/(2*0.22**2))

    def get_Sigma_1(self, t):
        Lambda_h = 1
        alpha_t = self.alpha_bar(t) / self.alpha_bar(t - self.delta_t)
        Sigma_1 = alpha_t / (1-alpha_t) * Lambda_h + 1 / (1 - self.alpha_bar(t - self.delta_t))
        # Sigma_1 = (1 - self.alpha_bar(t)) / (1 - self.alpha_bar(t - self.delta_t))
        return Sigma_1

    def get_sigma_0p5(self, t):
        Sigma_1 = self.get_Sigma_1(t)
        Sigma = 1 / Sigma_1
        return torch.sqrt(Sigma)

    def apply_sigma_0p5(self, x, t):
        Sigma0p5 = self.get_sigma_0p5(t)
        return Sigma0p5 * x

    def apply_sigma_1(self, x, t):
        Sigma_1 = self.get_Sigma_1(t)
        return Sigma_1 * x

    def apply_sigma(self, x, t):
        Sigma_1 = self.get_Sigma_1(t)
        Sigma = 1 / Sigma_1
        return Sigma * x

    def train_loss(self, model_out, x0, eps):
        if self.config.prediction_type == 'epsilon':
            return torch.nn.functional.mse_loss(model_out, eps)
        elif self.config.prediction_type == 'x0':
            return torch.nn.functional.mse_loss(model_out, x0)
        elif self.config.prediction_type == 'mu':
            pass
        elif self.config.prediction_type == 'x0+epsilon':
            return torch.nn.functional.mse_loss(model_out, torch.cat((x0, eps), dim=1))

    def get_xt(self, x0, t):
        mean = torch.sqrt(self.alpha_bar(t)) * x0
        eps = torch.randn_like(mean)
        return mean + torch.sqrt(1 - self.alpha_bar(t)) * eps, eps

    @torch.no_grad()
    def pred(self, model, xt, t):
        model_out = model(xt, self.config.time_scale * t)
        return model_out

    @torch.no_grad()
    
    def sample(self, model, batch_size=1):
        xt = torch.randn(batch_size, 3, self.config.img_size, self.config.img_size, dtype=torch.float32, device=self.device)

        for i, t in tqdm(enumerate(self.ts)):
            t = t.unsqueeze(0)

            model_out = model(xt, self.config.time_scale * t)
            # model_out = self.pred(model, xt, t)

            if t > self.delta_t:
                alpha_t = self.alpha_bar(t)/self.alpha_bar(t - self.delta_t)
                beta_t = 1 - alpha_t
                if self.config.prediction_type == 'epsilon':
                    xt = (xt  - (1 - alpha_t)/torch.sqrt(1 - self.alpha_bar(t)) * model_out)/torch.sqrt(alpha_t) + \
                         torch.sqrt((1 - self.alpha_bar(t - self.delta_t)) * beta_t/(1 - self.alpha_bar(t))) * torch.randn_like(model_out)
                elif self.config.prediction_type == 'x0':
                    xt = torch.sqrt(self.alpha_bar(t - self.delta_t)) * beta_t / (1 - self.alpha_bar(t)) * model_out + \
                        torch.sqrt(alpha_t) * (1 - self.alpha_bar(t - self.delta_t)) / (1 - self.alpha_bar(t)) * xt + \
                         torch.sqrt((1 - self.alpha_bar(t - self.delta_t)) * beta_t / (1 - self.alpha_bar(t))) * torch.randn_like(model_out)
                elif self.config.prediction_type == 'x0+epsilon':
                    x0_hat_direct = model_out[:, :3]
                    x0_hat_eps = (xt  - torch.sqrt(1 - self.alpha_bar(t)) * model_out[:, 3:])/torch.sqrt(self.alpha_bar(t))
                    x0_hat = x0_hat_eps#(1 - self.alpha_bar(t)) * x0_hat_direct + (self.alpha_bar(t)) * x0_hat_eps
                    xt = torch.sqrt(self.alpha_bar(t - self.delta_t)) * beta_t / (1 - self.alpha_bar(t)) * x0_hat + \
                        torch.sqrt(alpha_t) * (1 - self.alpha_bar(t - self.delta_t)) / (1 - self.alpha_bar(t)) * xt + \
                        torch.sqrt((1 - self.alpha_bar(t - self.delta_t)) * beta_t / (1 - self.alpha_bar(t))) * torch.randn_like(x0_hat)


            else:
                if self.config.prediction_type == 'epsilon':
                    xt = (xt  - (1 - alpha_t)/torch.sqrt(1 - self.alpha_bar(t)) * model_out)/torch.sqrt(alpha_t)
                elif self.config.prediction_type == 'x0':
                    xt = torch.sqrt(self.alpha_bar(t - self.delta_t)) * beta_t / (1 - self.alpha_bar(t)) * model_out + \
                        torch.sqrt(alpha_t) * (1 - self.alpha_bar(t - self.delta_t)) / (1 - self.alpha_bar(t)) * xt
                elif self.config.prediction_type == 'x0+epsilon':
                    x0_hat_direct = model_out[:, :3]
                    x0_hat_eps = (xt  - torch.sqrt(1 - self.alpha_bar(t)) * model_out[:, 3:])/torch.sqrt(self.alpha_bar(t))
                    x0_hat = (1 - self.alpha_bar(t)) * x0_hat_direct + (self.alpha_bar(t)) * x0_hat_eps
                    xt = x0_hat
                break

        return xt

# if __name__ == '__main__':
#     import argparse

#     args = argparse.ArgumentParser()
#     args.add_argument('--diffusion_steps', type=int, default=9)
#     args.add_argument('--diffusion_scale', type=int, default=2)
#     args.add_argument('--img_size', type=int, default=256)
#     args = args.parse_args()
#     args.device = torch.device("cuda:0")
#     temp = UDPM(config=args, device=args.device)
