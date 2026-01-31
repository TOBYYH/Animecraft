# FROM v5

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torchsummary
import time
from tqdm import tqdm
from thop import profile


class ResNetBlock(nn.Module):
    def __init__(self, c_num, k, groups) -> None:
        super().__init__()
        p = (k - 1) // 2
        self.layers = nn.Sequential(
            nn.GroupNorm(1, c_num),
            nn.Conv2d(c_num, c_num, k, 1, p, groups=groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_num, c_num, 1, 1, 0)
        )
    
    def forward(self, x):
        return self.layers(x) + x


class SelfAttention(nn.Module):
    def __init__(self, c_num) -> None:
        super().__init__()
        self.input = nn.GroupNorm(1, c_num),
        
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
    def __init__(self, in_c, out_c, k, groups, attention):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_c, out_c, 1, 1, 0),
            ResNetBlock(out_c, k, groups),
            ResNetBlock(out_c, k, groups)
        )
        if attention:
            self.attention = SelfAttention(out_c)
        else:
            self.attention = None
        self.out = ResNetBlock(out_c, k, groups)

    def forward(self, x):
        z = self.layers(x)
        if self.attention is not None:
            z = self.attention(z)
        return self.out(z)


class UNetBlockUp(nn.Module):
    def __init__(self, in_c, out_c, k, groups, attention):
        super().__init__()
        self.up = nn.Sequential(
            nn.GroupNorm(1, in_c),
            nn.ConvTranspose2d(in_c, out_c, 2, 2, 0)
        )
        self.norm = nn.GroupNorm(1, out_c)
        self.conv = nn.Conv2d(out_c*2, out_c, 1, 1, 0)
        self.res = nn.Sequential(
            ResNetBlock(out_c, k, groups),
            ResNetBlock(out_c, k, groups)
        )
        if attention:
            self.attention = SelfAttention(out_c)
        else:
            self.attention = None
        self.out = ResNetBlock(out_c, k, groups)

    def forward(self, x1, x2):
        x = torch.cat((self.up(x1), self.norm(x2)), dim=1)
        z = self.res(self.conv(x))
        if self.attention is not None:
            z = self.attention(z)
        return self.out(z)


class AcSuperResolution(nn.Module):
    def __init__(self):
        super().__init__()
        print("AcSuperResolution initializing...")
        
        c = (32*2, 32*4, 64*4, 128*4)
        self.conv_in = nn.Sequential(
            nn.Conv2d(3, c[0], 7, 1, 3),
            ResNetBlock(c[0], 7, 1),
            ResNetBlock(c[0], 7, 1),
            ResNetBlock(c[0], 7, 1)
        )
        
        self.down1 = UNetBlockDown(c[0], c[1], 7, 1, False)
        self.down2 = UNetBlockDown(c[1], c[2], 7, 2, True)
        self.down3 = UNetBlockDown(c[2], c[3], 3, 1, True)
        self.up1 = UNetBlockUp(c[3], c[2], 7, 2, True)
        self.up2 = UNetBlockUp(c[2], c[1], 7, 1, False)
        self.up3 = UNetBlockUp(c[1], c[0], 7, 1, False)
        
        self.upsample = nn.Sequential(
            nn.GroupNorm(1, c[0]),
            nn.ConvTranspose2d(c[0], c[0]//2, 2, 2, 0),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(c[0]//2, c[0]//4, 2, 2, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[0]//4, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        z1 = self.conv_in(x)
        z2 = self.down1(z1)
        z3 = self.down2(z2)
        z = self.down3(z3)
        z = self.up1(z, z3)
        z = self.up2(z, z2)
        z = self.up3(z, z1)
        return self.upsample(z)
    
    def cp(self, x):
        return checkpoint(self.forward, x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        print("AutoEncoder initializing...")
        
        c = (32, 64, 128, 256)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c[0], 4, 2, 1),
            ResNetBlock(c[0], 5, 1),
            ResNetBlock(c[0], 5, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[0], c[1], 4, 2, 1),
            ResNetBlock(c[1], 5, 1),
            ResNetBlock(c[1], 5, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[1], c[2], 4, 2, 1),
            ResNetBlock(c[2], 5, 2),
            ResNetBlock(c[2], 5, 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[2], c[3], 4, 2, 1),
            ResNetBlock(c[3], 5, 4),
            ResNetBlock(c[3], 5, 4),
            nn.SiLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c[3], c[2], 2, 2, 0),
            ResNetBlock(c[2], 5, 2),
            ResNetBlock(c[2], 5, 2),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(c[2], c[1], 2, 2, 0),
            ResNetBlock(c[1], 5, 1),
            ResNetBlock(c[1], 5, 1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(c[1], c[0], 2, 2, 0),
            ResNetBlock(c[0], 5, 1),
            ResNetBlock(c[0], 5, 1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(c[0], 3, 2, 2, 0),
            nn.Tanh()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        return self.decode(self.encode(x))


if __name__ == '__main__':
    H, W = 104, 184
    device = "cpu"
    model = AcSuperResolution()
    model.eval()
    model.to(device)
    torchsummary.summary(model, input_size=(3, H, W), device=device)
    z = torch.randn([1, 3, H, W], dtype=torch.float32, device=device)
    t = time.time()
    model.cp(z.detach())
    print(time.time() - t)
    flops, params = profile(model, inputs=[z.detach()])
    print(f"FLOPs: {flops / 1000.**3}G, params: {params / 1000.**2}M")
    
    H, W = 104*4, 184*4
    model = AutoEncoder()
    model.eval()
    model.to(device)
    torchsummary.summary(model, input_size=(3, H, W), device=device)
    z = torch.randn([1, 3, H, W], dtype=torch.float32, device=device)
    t = time.time()
    model(z.detach())
    print(time.time() - t)
    flops, params = profile(model, inputs=[z.detach()])
    print(f"FLOPs: {flops / 1000.**3}G, params: {params / 1000.**2}M")
