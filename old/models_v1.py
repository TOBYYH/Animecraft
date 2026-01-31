# V0: + group_norm

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torchsummary
import time
from tqdm import tqdm
from thop import profile

from utils import group_norm


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            group_norm(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            group_norm(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetBlockDown(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_c, out_c)
        )
        self.layer_emb = nn.Linear(time_dim, out_c)

    def forward(self, x, t):
        z = self.down(x)
        N, H, W = z.shape[0], z.shape[2], z.shape[3]
        emb = self.layer_emb(t)[:, :, None, None].repeat(N, 1, H, W)
        return z + emb


class UNetBlockUp(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_c + out_c, out_c)
        self.layer_emb = nn.Linear(time_dim, out_c)

    def forward(self, x1, x2, t):
        z = self.up(x1)
        z = self.conv(torch.cat((z, x2), dim=1))
        N, H, W = z.shape[0], z.shape[2], z.shape[3]
        emb = self.layer_emb(t)[:, :, None, None].repeat(N, 1, H, W)
        return z + emb


class AcDiffusionUNet(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        print("AcDiffusionUNetV1 initializing...")
        self.time_dim = time_dim
        
        self.time = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ELU(inplace=True)
        )
        c = (32*4, 32*9, 64*9, 128*10)
        self.conv_in = ConvBlock(3, c[0])
        
        self.down1 = UNetBlockDown(c[0], c[1], time_dim)
        self.down2 = UNetBlockDown(c[1], c[2], time_dim)
        self.down3 = UNetBlockDown(c[2], c[3], time_dim)
        self.up1 = UNetBlockUp(c[3], c[2], time_dim)
        self.up2 = UNetBlockUp(c[2], c[1], time_dim)
        self.up3 = UNetBlockUp(c[1], c[0], time_dim)
        
        self.out = nn.Conv2d(c[0], 3, 1, 1, 0)

    def forward(self, x, t=None):
        if t is None:
            t = torch.zeros([1, self.time_dim], dtype=x.dtype, device=x.device)
        tt = self.time(t)
        z1 = self.conv_in(x)
        z2 = self.down1(z1, tt)
        z3 = self.down2(z2, tt)
        z = self.down3(z3, tt)
        z = self.up1(z, z3, tt)
        z = self.up2(z, z2, tt)
        z = self.up3(z, z1, tt)
        return self.out(z)
    
    def cp(self, x, t=None):
        return checkpoint(self.forward, x, t)


def summary_UNet():
    H, W = 104, 184
    time_dim = 1024
    device = "cuda"
    model = AcDiffusionUNet(time_dim)
    model.eval()
    model.to(device)
    torchsummary.summary(model, input_size=(3, H, W), device=device)
    z = torch.randn([1, 3, H, W], dtype=torch.float32, device=device)
    t_emb = torch.randn([1, time_dim], dtype=torch.float32, device=device)
    looper = tqdm(range(100))
    t = time.time()
    for _ in looper:
        z = model.cp(z.detach(), t_emb)
    t = time.time() - t
    print(100. / t)
    flops, params = profile(model, inputs=(z.detach(), t_emb))
    print(f"FLOPs: {flops / 1000.**3}G, params: {params / 1000.**2}M")
    print(flops / 1000.**3 * 0.8 + params*2 / 1000.**2 * 0.1)
    print(params / 1000.**2 * 0.5 + t * 50.)


if __name__ == '__main__':
    summary_UNet()
