# V5: + ParallelConv2d

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torchsummary
import time
from tqdm import tqdm
from thop import profile

from utils import group_norm


class ParallelConv2d(nn.Module):
    def __init__(self, c_in, c_out, k1, k2, g1=1, g2=1) -> None:
        super().__init__()
        p1 = (k1 - 1) // 2
        p2 = (k2 - 1) // 2
        self.conv1 = nn.Conv2d(c_in, c_out, k1, 1, p1, groups=g1)
        self.conv2 = nn.Conv2d(c_in, c_out, k2, 1, p2, groups=g2)
    
    def forward(self, x):
        return self.conv1(x) + self.conv2(x)


class ResNetBlockDiff(nn.Module):
    def __init__(self, c_num, time_dim, k, groups) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            group_norm(c_num),
            ParallelConv2d(c_num, c_num, 1, k, 1, groups)
        )
        self.layer_emb = nn.Linear(time_dim, c_num)
        self.layer2 = nn.Sequential(
            nn.SiLU(inplace=True),
            ParallelConv2d(c_num, c_num, 1, k, 1, groups)
        )
    
    def forward(self, x, t):
        N, H, W = x.shape[0], x.shape[2], x.shape[3]
        tt = self.layer_emb(t)[:, :, None, None].repeat(N, 1, H, W)
        return self.layer2(self.layer1(x) + tt) + x


class SelfAttention(nn.Module):
    def __init__(self, c_num) -> None:
        super().__init__()
        self.input = group_norm(c_num)
        
        c = c_num // 2
        self.query = nn.Conv2d(c_num, c, 1, 1, 0)
        self.key = nn.Conv2d(c_num, c, 1, 1, 0)
        self.value = nn.Conv2d(c_num, c, 1, 1, 0)
        self.softmax  = nn.Softmax(dim=-1)
        
        self.out = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c_num, 1, 1, 0)
        )
    
    def forward(self, x):
        N, C, H, W = x.shape
        c = C // 2
        z = self.input(x)
        q = self.query(z).view(N, c, H * W)
        k = self.key(z).view(N, c, H * W)
        v = self.value(z).view(N, c, H * W)
        e = torch.bmm(q.permute(0, 2, 1), k)
        a = self.softmax(e)
        z = torch.bmm(v, a.permute(0, 2, 1)).view(N, c, H, W)
        return self.out(z) + x


class UNetBlockDown(nn.Module):
    def __init__(self, in_c, out_c, k, groups, time_dim, attention):
        super().__init__()
        self.down = nn.Sequential(
            group_norm(in_c),
            nn.Conv2d(in_c, out_c, 2, 2, 0)
        )
        self.res1 = ResNetBlockDiff(out_c, time_dim, k, groups)
        self.res2 = ResNetBlockDiff(out_c, time_dim, k, groups)
        if attention:
            self.attention = SelfAttention(out_c)
        else:
            self.attention = None

    def forward(self, x, t):
        z = self.res2(self.res1(self.down(x), t), t)
        if self.attention is not None:
            z = self.attention(z)
        return z


class UNetBlockUp(nn.Module):
    def __init__(self, in_c, out_c, k, groups, time_dim, attention):
        super().__init__()
        self.up = nn.Sequential(
            group_norm(in_c),
            nn.ConvTranspose2d(in_c, out_c, 2, 2, 0)
        )
        self.conv = nn.Conv2d(out_c * 2, out_c, 1, 1, 0)
        self.res1 = ResNetBlockDiff(out_c, time_dim, k, groups)
        self.res2 = ResNetBlockDiff(out_c, time_dim, k, groups)
        if attention:
            self.attention = SelfAttention(out_c)
        else:
            self.attention = None

    def forward(self, x1, x2, t):
        z = self.conv(torch.cat((self.up(x1), x2), dim=1))
        z = self.res2(self.res1(z, t), t)
        if self.attention is not None:
            z = self.attention(z)
        return z


class AcDiffusionUNet(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        print("AcDiffusionUNetV6 initializing...")
        self.time_dim = time_dim
        
        self.time = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(inplace=True)
        )
        c = (32*2, 32*5, 64*6, 128*6)
        self.conv_in = ParallelConv2d(3, c[0], 1, 7)
        self.res1 = ResNetBlockDiff(c[0], time_dim, 7, 1)
        self.res2 = ResNetBlockDiff(c[0], time_dim, 7, 1)
        
        self.down1 = UNetBlockDown(c[0], c[1], 7, 1, time_dim, False)
        self.down2 = UNetBlockDown(c[1], c[2], 7, 2, time_dim, True)
        self.down3 = UNetBlockDown(c[2], c[3], 7, 4, time_dim, True)
        self.up1 = UNetBlockUp(c[3], c[2], 7, 2, time_dim, True)
        self.up2 = UNetBlockUp(c[2], c[1], 7, 1, time_dim, False)
        self.up3 = UNetBlockUp(c[1], c[0], 7, 1, time_dim, False)
        
        self.out = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 3, 1, 1, 0)
        )

    def forward(self, x, t=None):
        if t is None:
            t = torch.zeros([1, self.time_dim], dtype=x.dtype, device=x.device)
        tt = self.time(t)
        z1 = self.res2(self.res1(self.conv_in(x), tt), tt)
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
