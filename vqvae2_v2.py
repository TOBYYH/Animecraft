import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from torchvision import utils

from utils import *


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResNetBlock(nn.Module):
    def __init__(self, in_c, c, act="LReLU") -> None:
        super().__init__()
        self.layers = nn.Sequential(
            group_norm(in_c),
            nn.Conv2d(in_c, c, 3, 1, 1),
            activition(act, True),
            nn.Conv2d(c, in_c, 3, 1, 1)
        )
    
    def forward(self, x):
        return self.layers(x) + x


class Encoder(nn.Module):
    def __init__(self, in_c, c, n_res, c_res, mode, act="LReLU"):
        super().__init__()
        
        if mode == 'b':
            blocks = [
                nn.Conv2d(in_c, c//8, 4, 2, 1),
                activition(act, True),
                nn.Conv2d(c//8, c//2, 4, 2, 1),
                ResNetBlock(c//2, c_res, act),
                activition(act, True),
                nn.Conv2d(c//2, c, 4, 2, 1)
            ]
        elif mode == 't':
            blocks = [
                nn.Conv2d(in_c, c, 4, 2, 1),
                ResNetBlock(c, c_res, act),
                activition(act, True),
                nn.Conv2d(c, c, 4, 2, 1)
            ]
        
        for _ in range(n_res):
            blocks.append(ResNetBlock(c, c_res, act))
        
        blocks.append(activition(act, True))
        
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.blocks(x)


class Decoder(nn.Module):
    def __init__(self, in_c, out_c, c, n_res, c_res, mode, act="LReLU"):
        super().__init__()
        
        blocks = [nn.Conv2d(in_c, c, 3, 1, 1)]
        for _ in range(n_res):
            blocks.append(ResNetBlock(c, c_res, act))
        blocks.append(activition(act, True))
        
        if mode == 'b':
            blocks.extend([
                nn.ConvTranspose2d(c, c//2, 4, 2, 1),
                ResNetBlock(c//2, c_res, act),
                activition(act, True),
                nn.ConvTranspose2d(c//2, c//8, 4, 2, 1),
                activition(act, True),
                nn.ConvTranspose2d(c//8, out_c, 4, 2, 1)
            ])
        elif mode == 't':
            blocks.extend([
                nn.ConvTranspose2d(c, c, 4, 2, 1),
                ResNetBlock(c, c_res, act),
                activition(act, True),
                nn.ConvTranspose2d(c, out_c, 4, 2, 1)
            ])
        
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.blocks(x)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_c=3,
        c=24*8,
        n_res=2,
        c_res=32,
        embed_dim=64,
        n_embed=512,
        act="tanh"
    ):
        super().__init__()

        self.enc_b = Encoder(in_c, c, n_res, c_res, 'b', act)
        self.enc_t = Encoder(c, c, n_res, c_res, 't', act)
        self.quantize_conv_t = nn.Conv2d(c, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(embed_dim, embed_dim, c, n_res, c_res, 't', act)
        self.quantize_conv_b = nn.Conv2d(embed_dim + c, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1),
            activition(act, True),
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        )
        self.dec = Decoder(embed_dim*2, in_c, c, n_res, c_res, 'b', act)

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


def summary():
    H, W = 9*64, 16*64
    device = "cuda"
    model = VQVAE().eval().to(device)
    # torchsummary.summary(model, input_size=(3, H, W), device=device)
    x = torch.randn([1, 3, H, W], dtype=torch.float32, device=device)
    t = time.time()
    for _ in tqdm(range(100)):
        x, _ = model(x.detach())
    t = time.time() - t
    print(100. / t)
    quant_t, quant_b, _, _, _ = model.encode(x)
    print(quant_t.shape, quant_b.shape)


def train():
    epoch = 100
    size = 9*64, 16*64
    path = "../dataset/LL/*"
    device = "cuda"
    dataset = AcDatasetV2(path, size)
    loader = DataLoader(dataset, batch_size=24, shuffle=True, num_workers=8, drop_last=True)
    model = VQVAE().train().to(device)
    load_model_file(model, "AcVQVAE2.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4,
                                                    total_steps=epoch*len(loader),
                                                    div_factor=10.,
                                                    final_div_factor=1.)
    loss_func = nn.MSELoss()
    n = 0
    e = 1
    tt = time.time()
    for e in range(1, epoch + 1):
        looper = tqdm(loader, total=len(loader))
        for data in looper:
            image = data.to(device)
            
            optimizer.zero_grad()
            out, latent_loss = model(image.detach())
            loss = loss_func(out, image) + latent_loss.mean() * 0.25
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            n += out.shape[0]
            looper.set_description(f"epoch: {e}")
            looper.set_postfix(loss=loss.item(), sample_num=n, lr=optimizer.param_groups[0]['lr'])
        
        if e % 5 == 0:
            torch.save(model.state_dict(), f"models/AcVQVAE2-epoch{e}.pt")
            
            model.eval()
            if image.shape[0] > 5:
                image = image[:5]
            with torch.no_grad():
                out, _ = model(image)
            utils.save_image(
                torch.cat([image, out], 0),
                f"samples/epoch-{e}.png",
                nrow=image.shape[0],
                normalize=True
            )
            model.train()
    
    print(f"Train time: {time.time() - tt}")


if __name__ == '__main__':
    # summary()
    train()
