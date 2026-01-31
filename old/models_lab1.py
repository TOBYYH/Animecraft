import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torchsummary
import time
from tqdm import tqdm
from thop import profile

from utils import group_norm, activition


# class ParallelConv2d(nn.Module):
#     def __init__(self, c_num, k, groups) -> None:
#         super().__init__()
#         p = (k - 1) // 2
#         self.conv1x1 = nn.Conv2d(c_num, c_num, 1, 1, 0)
#         self.conv = nn.Sequential(
#             nn.Conv2d(c_num, c_num, k, 1, p, groups=groups),
#             nn.Conv2d(c_num, c_num, 1, 1, 0)
#         )
    
#     def forward(self, x):
#         return self.conv1x1(x) + self.conv(x)


class ResNetBlock(nn.Module):
    def __init__(self, c_num, k, groups) -> None:
        super().__init__()
        p = (k - 1) // 2
        self.layers = nn.Sequential(
            group_norm(c_num),
            nn.Conv2d(c_num, c_num, k, 1, p, groups=groups),
            activition("atan"),
            nn.Conv2d(c_num, c_num, 1, 1, 0)
        )
    
    def forward(self, x):
        return self.layers(x) + x
        # return checkpoint(self.layers, x) + x


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        print("AutoEncoder initializing...")
        
        a = "atan"
        c = (32, 64, 128, 256)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c[0], 4, 2, 1),
            ResNetBlock(c[0], 5, 1),
            ResNetBlock(c[0], 5, 1),
            activition(a),
            nn.Conv2d(c[0], c[1], 4, 2, 1),
            ResNetBlock(c[1], 5, 1),
            ResNetBlock(c[1], 5, 1),
            activition(a),
            nn.Conv2d(c[1], c[2], 4, 2, 1),
            ResNetBlock(c[2], 5, 1),
            ResNetBlock(c[2], 5, 1),
            activition(a),
            nn.Conv2d(c[2], c[3], 4, 2, 1),
            ResNetBlock(c[3], 5, 1),
            ResNetBlock(c[3], 5, 1),
            activition(a)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c[3], c[2], 2, 2, 0),
            ResNetBlock(c[2], 5, 1),
            ResNetBlock(c[2], 5, 1),
            activition(a),
            nn.ConvTranspose2d(c[2], c[1], 2, 2, 0),
            ResNetBlock(c[1], 5, 1),
            ResNetBlock(c[1], 5, 1),
            activition(a),
            nn.ConvTranspose2d(c[1], c[0], 2, 2, 0),
            ResNetBlock(c[0], 5, 1),
            ResNetBlock(c[0], 5, 1),
            activition(a),
            nn.ConvTranspose2d(c[0], 3, 2, 2, 0),
            nn.Tanh()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == '__main__':
    device = "cpu"
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
