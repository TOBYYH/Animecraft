import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torchsummary
import time
from tqdm import tqdm
from thop import profile
import matplotlib.pylab as plt

from utils import group_norm


class ResNetBlock(nn.Module):
    def __init__(self, c_num, k, groups) -> None:
        super().__init__()
        p = (k - 1) // 2
        self.layers = nn.Sequential(
            group_norm(c_num),
            nn.Conv2d(c_num, c_num, k, 1, p, groups=groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_num, c_num, 1, 1, 0)
        )
    
    def forward(self, x):
        return self.layers(x) + x


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


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        c = (32, 64, 128)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c[0], 7, 1, 3),
            ResNetBlock(c[0], 7, 1),
            ResNetBlock(c[0], 7, 1),
            ResNetBlock(c[0], 7, 1),
            group_norm(c[0]),
            nn.Conv2d(c[0], c[1], 2, 2, 0),
            ResNetBlock(c[1], 7, 1),
            ResNetBlock(c[1], 7, 1),
            SelfAttention(c[1]),
            ResNetBlock(c[1], 7, 1),
            group_norm(c[1]),
            nn.Conv2d(c[1], c[2], 2, 2, 0),
            ResNetBlock(c[2], 7, 1),
            ResNetBlock(c[2], 7, 1),
            SelfAttention(c[2]),
            ResNetBlock(c[2], 7, 1)
        )
        
        self.beta = 0.25
        self.codebook = nn.Embedding(c[2], c[2])
        
        self.decoder = nn.Sequential(
            ResNetBlock(c[2], 7, 1),
            ResNetBlock(c[2], 7, 1),
            SelfAttention(c[2]),
            ResNetBlock(c[2], 7, 1),
            group_norm(c[2]),
            nn.ConvTranspose2d(c[2], c[1], 2, 2, 0),
            ResNetBlock(c[1], 7, 1),
            ResNetBlock(c[1], 7, 1),
            SelfAttention(c[1]),
            ResNetBlock(c[1], 7, 1),
            group_norm(c[1]),
            nn.ConvTranspose2d(c[1], c[0], 2, 2, 0),
            ResNetBlock(c[0], 7, 1),
            ResNetBlock(c[0], 7, 1),
            ResNetBlock(c[0], 7, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c[0], 3, 1, 1, 0)
        )
    
    def vq(self, z):
        N, C, H, W = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()
        _z = z.view(-1, C)
        d = (_z.unsqueeze(1) - self.codebook.weight.unsqueeze(0)) ** 2
        d = d.sum(-1)
        ids = torch.argmin(d, dim=1).unsqueeze(1)
        one_hot = torch.zeros_like(d, dtype=torch.float32, device=z.device)
        one_hot.scatter_(1, ids, 1)
        qz = torch.matmul(one_hot, self.codebook.weight).view(N, H, W, C)
        loss1 = F.mse_loss(qz.detach(), z)
        loss2 = F.mse_loss(qz, z.detach())
        vq_loss = loss1 * self.beta + loss2
        qz = z + (qz - z).detach()
        return qz.permute(0, 3, 1, 2).contiguous(), vq_loss
    
    def forward(self, x):
        z, vq_loss = self.vq(self.encoder(x))
        return self.decoder(z), vq_loss
    
    def plot_codebook(self):
        codes = self.codebook.weight.detach().cpu().numpy()
        for i in range(codes.shape[0]):
            plt.scatter(codes[i, 0], codes[i, 1])
        plt.show()


if __name__ == '__main__':
    times = 20
    H, W = 104, 184
    device = "cuda"
    model = VQVAE()
    model.eval()
    model.to(device)
    torchsummary.summary(model, input_size=(3, H, W), device=device)
    x = torch.randn([1, 3, H, W], dtype=torch.float32, device=device)
    z = model.encoder(x)
    print(z.shape)
    print(model.vq(z)[0].shape)
    model.plot_codebook()
