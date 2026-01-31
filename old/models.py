from torch import nn
import torchsummary
from torch.utils.checkpoint import checkpoint
import time

from utils import *


def group_norm(channel:int) -> nn.GroupNorm:
    if channel <= 8:
        return nn.GroupNorm(1, channel)
    elif channel <= 32:
        return nn.GroupNorm(4, channel)
    elif channel <= 128:
        return nn.GroupNorm(8, channel)
    elif channel <= 1024:
        return nn.GroupNorm(32, channel)
    else:
        raise NotImplementedError(channel)


class Conv2dWithGroupConv(nn.Module):
    def __init__(self, c_in, c_out, gc_k, gc_s, gc_p, groups) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.gc = nn.Sequential(
            nn.Conv2d(c_in, c_in, gc_k, gc_s, gc_p, groups=groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_in, c_out, 1, 1, 0)
        ) if groups > 1 else nn.Conv2d(c_in, c_out, gc_k, gc_s, gc_p)
    
    def forward(self, x):
        return self.conv(x) + self.gc(x)


class ResNetBlockDiff(nn.Module):
    def __init__(self, c_num, time_c, gc_k, groups) -> None:
        super().__init__()
        assert gc_k % 2 == 1
        gc_p = (gc_k - 1) // 2
        self.layer1 = nn.Sequential(
            group_norm(c_num),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_num, c_num, 3, 1, 1)
        )
        self.t_emb = nn.Linear(time_c, c_num)
        self.layer2 = nn.Sequential(
            nn.SiLU(inplace=True),
            Conv2dWithGroupConv(c_num, c_num, gc_k, 1, gc_p, groups),
            group_norm(c_num),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_num, c_num, 3, 1, 1)
        )
    
    def forward(self, x, t):
        tt = self.t_emb(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return self.layer2(self.layer1(x) + tt) + x


class SelfAttention(nn.Module):
    def __init__(self, c_num) -> None:
        super().__init__()
        self.input = nn.Sequential(
            group_norm(c_num),
            nn.SiLU(inplace=True)
        )
        
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
        f = self.input(x)
        q = self.query(f).view(N, c, H * W)
        k = self.key(f).view(N, c, H * W)
        v = self.value(f).view(N, c, H * W)
        e = torch.bmm(q.permute(0, 2, 1), k)
        a = self.softmax(e)
        f = torch.bmm(v, a.permute(0, 2, 1)).view(N, c, H, W)
        return self.out(f) + x


class UNetBlockDown(nn.Module):
    def  __init__(self, in_c, out_c, gc_k, groups, time_c, attention) -> None:
        super().__init__()
        self.conv_down = nn.Sequential(
            group_norm(in_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_c, out_c, 2, 2, 0)
        )
        self.res1 = ResNetBlockDiff(out_c, time_c, gc_k, groups)
        if attention:
            self.attention = SelfAttention(out_c)
        else:
            self.attention = None
        self.res2 = ResNetBlockDiff(out_c, time_c, gc_k, groups)
    
    def forward(self, x, t):
        f = self.conv_down(x)
        f = self.res1(f, t)
        if self.attention is not None:
            f = self.attention(f)
        f = self.res2(f, t)
        return f


class UNetBlockUp(nn.Module):
    def  __init__(self, in_c, out_c, gc_k, groups, time_c, attention) -> None:
        super().__init__()
        self.conv_up = nn.Sequential(
            group_norm(in_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_c, out_c * 4, 1, 1, 0),
            nn.PixelShuffle(2)
        )
        self.conv_cat = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c * 2, out_c, 1, 1, 0)
        )
        self.res1 = ResNetBlockDiff(out_c, time_c, gc_k, groups)
        if attention:
            self.attention = SelfAttention(out_c)
        else:
            self.attention = None
        self.res2 = ResNetBlockDiff(out_c, time_c, gc_k, groups)
    
    def forward(self, x1, x2, t):
        f = self.conv_up(x1)
        f = self.conv_cat(torch.cat((f, x2), dim=1))
        f = self.res1(f, t)
        if self.attention is not None:
            f = self.attention(f)
        f = self.res2(f, t)
        return f


class AcDiffusionUNet(nn.Module):
    def __init__(self, time_dim) -> None:
        super().__init__()
        self.time_dim = time_dim
        
        self.time_emb = nn.SiLU()
        self.conv_in = nn.Sequential(
            Conv2dWithGroupConv(3, 64, 11, 1, 5, 1),
            nn.SiLU(inplace=True),
            Conv2dWithGroupConv(64, 64, 9, 1, 4, 1)
        ) # (*, 64, 128, 128)
        self.res_in = ResNetBlockDiff(64, time_dim, 9, 1) # (*, 64, 128, 128)
        
        self.down1 = UNetBlockDown(64, 128, 5, 1, time_dim, True) # (*, 128, 64, 64)
        self.down2 = UNetBlockDown(128, 256, 5, 2, time_dim, True) # (*, 256, 32, 32)
        self.down3 = UNetBlockDown(256, 512, 5, 4, time_dim, True) # (*, 512, 16, 16)
        self.up1 = UNetBlockUp(512, 256, 5, 2, time_dim, True) # (*, 256, 32, 32)
        self.up2 = UNetBlockUp(256, 128, 5, 1, time_dim, True) # (*, 128, 64, 64)
        self.up3 = UNetBlockUp(128, 64, 9, 1, time_dim, False) # (*, 64, 128, 128)
        
        self.out = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 3, 1, 1, 0)
        ) # (*, 3, 128, 128)
    
    def forward(self, x, t=None):
        if t is None:
            t = torch.zeros([x.shape[0], self.time_dim], dtype=x.dtype, device=x.device)
        t_emb = self.time_emb(t)
        f = self.conv_in(x)
        f1 = self.res_in(f, t_emb)
        f2 = self.down1(f1, t_emb)
        f3 = self.down2(f2, t_emb)
        f = self.down3(f3, t_emb)
        f = self.up1(f, f3, t_emb)
        f = self.up2(f, f2, t_emb)
        f = self.up3(f, f1, t_emb)
        return self.out(f)
    
    def cp(self, x, t=None):
        x.requires_grad = True
        return checkpoint(self.forward, x, t)


def summary_U():
    H, W = 104, 184
    step_num = 1000
    time_dim = 1024
    diff_pars = (0.95, 0.0001, 0.05)
    device = "cuda"
    model = AcDiffusionUNet(1024)
    model.eval()
    model.to(device)
    torchsummary.summary(model, input_size=(3, H, W), device=device)


class ResNetBlockSR(nn.Module):
    def __init__(self, c_num, gc_k, groups) -> None:
        super().__init__()
        assert gc_k % 2 == 1
        gc_p = (gc_k - 1) // 2
        self.layers = nn.Sequential(
            group_norm(c_num),
            nn.SiLU(inplace=True),
            Conv2dWithGroupConv(c_num, c_num, gc_k, 1, gc_p, groups),
            group_norm(c_num),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_num, c_num, 3, 1, 1)
        )
    
    def forward(self, x):
        return self.layers(x) + x


class SRBlockDown(nn.Module):
    def  __init__(self, in_c, out_c, gc_k, groups, attention) -> None:
        super().__init__()
        self.down = nn.Sequential(
            group_norm(in_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_c, out_c, 1, 1, 0),
            nn.MaxPool2d(2)
        )
        self.res1 = ResNetBlockSR(out_c, gc_k, groups)
        if attention:
            self.attention = SelfAttention(out_c)
        else:
            self.attention = None
        self.res2 = ResNetBlockSR(out_c, gc_k, groups)
    
    def forward(self, x):
        f = self.down(x)
        f = self.res1(f)
        if self.attention is not None:
            f = self.attention(f)
        f = self.res2(f)
        return f


class SRBlockUp(nn.Module):
    def  __init__(self, in_c, out_c, gc_k, groups, attention) -> None:
        super().__init__()
        self.up = nn.Sequential(
            group_norm(in_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_c, out_c * 4, 1, 1, 0),
            nn.PixelShuffle(2)
        )
        self.conv_cat = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c * 2, out_c, 1, 1, 0)
        )
        self.res1 = ResNetBlockSR(out_c, gc_k, groups)
        if attention:
            self.attention = SelfAttention(out_c)
        else:
            self.attention = None
        self.res2 = ResNetBlockSR(out_c, gc_k, groups)
    
    def forward(self, x1, x2):
        f = self.up(x1)
        f = self.conv_cat(torch.cat((f, x2), dim=1))
        f = self.res1(f)
        if self.attention is not None:
            f = self.attention(f)
        f = self.res2(f)
        return f


class AcSuperResolution(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        c = (64, 128, 256, 512)
        self.block_in = nn.Sequential(
            Conv2dWithGroupConv(3, c[0], 11, 1, 5, 1),
            nn.SiLU(inplace=True),
            Conv2dWithGroupConv(c[0], c[0], 11, 1, 5, 1),
            ResNetBlockSR(c[0], 11, 1),
            ResNetBlockSR(c[0], 11, 1)
        )
        
        self.down1 = SRBlockDown(c[0], c[1], 9, 1, True)
        self.down2 = SRBlockDown(c[1], c[2], 7, 1, True)
        self.down3 = SRBlockDown(c[2], c[3], 5, 1, True)
        self.up1 = SRBlockUp(c[3], c[2], 7, 1, True)
        self.up2 = SRBlockUp(c[2], c[1], 9, 1, True)
        self.up3 = SRBlockUp(c[1], c[0], 11, 1, False)
        
        c = (c[0], 256, 128)
        self.upsample = nn.Sequential(
            nn.SiLU(inplace=True),
            Conv2dWithGroupConv(c[0], c[1], 7, 1, 3, 1),
            nn.PixelShuffle(2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[1] // 4, c[2], 3, 1, 1),
            nn.PixelShuffle(2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[2] // 4, 3, 1, 1, 0),
            nn.Tanh()
        )
    
    def forward(self, x):
        f1 = self.block_in(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f = self.down3(f3)
        f = self.up1(f, f3)
        f = self.up2(f, f2)
        f = self.up3(f, f1)
        return self.upsample(f)


def summary_SR():
    times = 20
    H, W = 104, 184
    batch_size = 4
    cp = False
    device = "cuda"
    model = AcSuperResolution()
    model.to(device)
    torchsummary.summary(model, input_size=(3, H, W), device=device)
    x = torch.randn([batch_size, 3, H, W], dtype=torch.float32, device=device)
    print("Warming up...")
    for _ in range(10):
        y = model(x).detach()
    print("Testing...")
    t = time.time()
    for _ in range(times):
        y = model(x).detach()
    print(f"Predicting time: {(time.time() - t) / times}")


class AcDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        c = (16, 32, 64, 96, 128, 256, 384, 512)
        self.layers = nn.Sequential(
            Conv2dWithGroupConv(3, c[0], 11, 1, 5, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[0], c[1], 4, 2, 1),
            ResNetBlockSR(c[1], 9, 1),
            ResNetBlockSR(c[1], 9, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[1], c[2], 4, 2, 1),
            ResNetBlockSR(c[2], 7, 1),
            ResNetBlockSR(c[2], 7, 1),
            group_norm(c[2]),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[2], c[3], 4, 2, 1),
            ResNetBlockSR(c[3], 5, 1),
            ResNetBlockSR(c[3], 5, 1),
            group_norm(c[3]),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[3], c[4], 4, 2, 1),
            ResNetBlockSR(c[4], 5, 2),
            ResNetBlockSR(c[4], 5, 2),
            group_norm(c[4]),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[4], c[5], 4, 2, 1),
            ResNetBlockSR(c[5], 5, 4),
            ResNetBlockSR(c[5], 5, 4),
            group_norm(c[5]),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[5], c[6], 3, 2, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[6], c[6], 3, 1, 1),
            group_norm(c[6]),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[6], c[7], (2, 3), 2, 0),
            nn.Flatten(),
            nn.Linear(512*3*5, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, cp=False):
        assert x.shape[-2:] == (104*4, 184*4)
        if cp:
            return checkpoint(self.layers, x)
        else:
            return self.layers(x)


def summary_D():
    times = 20
    H, W = 104*4, 184*4
    batch_size = 4
    cp = False
    device = "cuda"
    model = AcDiscriminator()
    model.to(device)
    torchsummary.summary(model, input_size=(3, H, W), device=device)
    x = torch.randn([batch_size, 3, H, W], dtype=torch.float32, device=device)
    print("Warming up...")
    for _ in range(10):
        y = model(x, cp).detach()
    print("Testing...")
    t = time.time()
    for _ in range(times):
        y = model(x, cp).detach()
    print(f"Predicting time: {(time.time() - t) / times}")


if __name__ == '__main__':
    summary_U()
    # summary_SR()
    # summary_D()
