import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from torchvision import utils
import torchinfo

from utils import *


class ResBlock1(nn.Module):
    def __init__(self, c, act="tanh") -> None:
        super().__init__()
        self.nn = nn.Sequential(
            group_norm(c),
            nn.Conv2d(c, c, 3, 1, 1),
            activition(act, True),
            nn.Conv2d(c, c, 3, 1, 1)
        )
    
    def forward(self, x):
        return x + self.nn(x)


class Discriminator1(nn.Module):
    def __init__(self, act="ELU"):
        super().__init__()
        print("Discriminator initializing...")
        
        C_IN = 32
        S = 64
        C = (S, S*2, S*4)
        self.layers = nn.Sequential(
            nn.Conv2d(3, C[0], 8, 4, 2),
            ResBlock1(C[0], act),
            nn.Conv2d(C[0], C[1], 4, 2, 1),
            ResBlock1(C[1], act),
            nn.Conv2d(C[1], C[2], 4, 2, 1),
            ResBlock1(C[2], act),
            nn.Conv2d(C[2], 1, 1, 1, 0),
            activition("sigmoid", True)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def summary(self):
        H, W = 9*64, 16*64
        device, dtype = "cuda", torch.float32
        self.eval().to(device)
        torchinfo.summary(self, input_size=(1, 3, H, W), device=device)
        self.eval().to(device, dtype)
        z = torch.randn([1, 3, H, W], dtype=dtype, device=device)
        looper = tqdm(range(100))
        t = time.time()
        for _ in looper:
            o = self(z.detach())
        t = time.time() - t
        print(f"{100. / t} /s {o.shape}")


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
    def __init__(self, c_num, k, groups, act="LReLU") -> None:
        super().__init__()
        p = (k - 1) // 2
        self.layers = nn.Sequential(
            group_norm(c_num),
            nn.Conv2d(c_num, c_num, k, 1, p, groups=groups),
            activition(act),
            nn.Conv2d(c_num, c_num, 1, 1, 0)
        )
    
    def forward(self, x):
        return self.layers(x) + x


class SelfAttention(nn.Module):
    def __init__(self, c_num) -> None:
        super().__init__()
        self.ln = nn.LayerNorm([c_num])
        self.attention = nn.MultiheadAttention(c_num, 4, batch_first=True)
    
    def forward(self, x):
        N, C, H, W = x.shape
        x = x.view(N, C, H*W).swapaxes(1, 2)
        x_ln = self.ln(x)
        a, _  = self.attention(x_ln, x_ln, x_ln)
        x = x + a
        return x.swapaxes(2, 1).view(N, C, H, W)


def res_block(res_n, a_id, c, k, g, act):
    blocks = []
    for i in range(res_n):
        blocks.append(ResNetBlock(c, k, g, act))
        if i == a_id:
            blocks.append(SelfAttention(c))
    return blocks


class VQVAE(nn.Module):
    def __init__(self, act="ELU"):
        super().__init__()
        print("VQVAE initializing...")
        
        S = 80
        C = (S, S*2, S*4)
        K = (5, 5, 3)
        G = (1, 1, 1)
        A = (-1, -1, 1)
        RES_N = 3
        EMB_DIM = 64
        EMB_NUM = 1024
        
        blocks = [
            nn.Conv2d(3, S//4, 4, 2, 1),
            activition(act),
            nn.Conv2d(S//4, C[0], 4, 2, 1)
        ]
        for i in range(len(C)):
            if i > 0:
                blocks.append(nn.Conv2d(C[i-1], C[i], 4, 2, 1))
            blocks += res_block(RES_N, A[i], C[i], K[i], G[i], act)
            blocks.append(activition(act))
        self.enc_b = nn.Sequential(*blocks)
        
        blocks = [nn.Conv2d(C[-1], C[-1], 4, 2, 1)]
        blocks += res_block(RES_N, A[-1], C[-1], 3, 1, act)
        blocks += [
            activition(act),
            nn.Conv2d(C[-1], C[-1], 4, 2, 1)
        ]
        blocks += res_block(RES_N, A[-1], C[-1], 3, 1, act)
        blocks.append(activition(act))
        self.enc_t = nn.Sequential(*blocks)
        
        self.quantize_conv_t = nn.Conv2d(C[-1], EMB_DIM, 1, 1, 0)
        self.quantize_t = Quantize(EMB_DIM, EMB_NUM)
        
        blocks = [nn.Conv2d(EMB_DIM, C[-1], 1, 1, 0)]
        blocks += res_block(RES_N, A[-1], C[-1], 3, 1, act)
        blocks += [
            activition(act),
            nn.ConvTranspose2d(C[-1], C[-1], 4, 2, 1)
        ]
        blocks += res_block(RES_N, A[-1], C[-1], 3, 1, act)
        blocks += [
            activition(act),
            nn.ConvTranspose2d(C[-1], EMB_DIM, 4, 2, 1)
        ]
        self.dec_t = nn.Sequential(*blocks)
        
        self.quantize_conv_b = nn.Conv2d(EMB_DIM + C[-1], EMB_DIM, 1, 1, 0)
        self.quantize_b = Quantize(EMB_DIM, EMB_NUM)
        self.upsample_t = nn.Sequential(
            nn.ConvTranspose2d(EMB_DIM, EMB_DIM, 4, 2, 1),
            activition(act),
            nn.ConvTranspose2d(EMB_DIM, EMB_DIM, 4, 2, 1)
        )
        
        blocks = [nn.Conv2d(EMB_DIM * 2, C[-1], 1, 1, 0)]
        for i in range(len(C) - 1, -1, -1):
            blocks += res_block(RES_N, A[i], C[i], K[i], G[i], act)
            if i > 0:
                blocks += [
                    group_norm(C[i]),
                    nn.ConvTranspose2d(C[i], C[i-1], 4, 2, 1)
                ]
        blocks += [
            group_norm(C[0]),
            nn.ConvTranspose2d(C[0], S//4, 4, 2, 1),
            activition(act),
            nn.ConvTranspose2d(S//4, 3, 4, 2, 1),
            nn.Tanh()
        ]
        self.dec = nn.Sequential(*blocks)
    
    def forward(self, x):
        quant_t, quant_b, diff, _, _ = self.encode(x)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, x):
        enc_b = self.enc_b(x)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], dim=1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], dim=1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
    
    def summary(self):
        H, W = 9*64, 16*64
        device, dtype = "cuda", torch.float32
        self.eval().to(device)
        torchinfo.summary(self, input_size=(1, 3, H, W), device=device)
        self.eval().to(device, dtype)
        z = torch.randn([1, 3, H, W], dtype=dtype, device=device)
        quant_t, quant_b, _, _, _ = self.encode(z)
        print(quant_b.shape, quant_t.shape)
        looper = tqdm(range(100))
        t = time.time()
        for _ in looper:
            z, _ = self(z.detach())
        t = time.time() - t
        print(f"{100. / t} /s")


def train(name, model_G, model_D, epoch, train_G=True, train_D=False, loss_plot=True, bs_G=6, bs_D=16):
    assert train_G or train_D
    save_step = 10
    if train_G:
        batch_size = bs_G
    else:
        batch_size = bs_D
    H, W = 9*64, 16*64
    path = "frames/*.png"
    device, dtype = "cuda", torch.float32
    if epoch < save_step:
        save_step = epoch
    epoch_exist = len(glob.glob(f"samples/{name}/epoch*")) * save_step
    dataset = AcDataset(path, (H, W), dtype)
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, drop_last=True)
    if train_G:
        model_G.train().to(device, dtype)
        optimizer_G = torch.optim.AdamW(model_G.parameters(), lr=1e-5)
        scheduler_G = torch.optim.lr_scheduler.OneCycleLR(optimizer_G, max_lr=2e-5,
                                                          total_steps=epoch*len(loader),
                                                          div_factor=10.0,
                                                          final_div_factor=1.0)
    else:
        model_G.eval().to(device, dtype)
    load_model_file(model_G, f"{name}.pt")
    
    if train_D:
        model_D.train().to(device, dtype)
        load_model_file(model_D, f"{name}-D.pt")
        optimizer_D = torch.optim.AdamW(model_D.parameters(), lr=1e-5)
        scheduler_D = torch.optim.lr_scheduler.OneCycleLR(optimizer_D, max_lr=2e-5,
                                                          total_steps=epoch*len(loader),
                                                          div_factor=10.0,
                                                          final_div_factor=1.0)
    recon_loss_list = []
    A_loss_list = []
    D_loss_list = []
    latent_weight = 0.2
    gan_weight = 0.005
    A_loss = 0.0
    D_loss = 0.0
    c = 0
    tt = time.time()
    for e in range(1, epoch + 1):
        looper = tqdm(loader, total=len(loader))
        for data in looper:
            images = data.to(device).requires_grad_(False)
            
            if train_G:
                optimizer_G.zero_grad()
                out, latent_loss = model_G(images)
                recon_loss = F.mse_loss(out, images)
                loss = recon_loss + latent_loss.mean() * latent_weight
                if train_D and D_loss < 0.4:
                    fake_pred = model_D(out)
                    loss_A = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
                    A_loss = loss_A.item()
                    loss += loss_A * gan_weight
                loss.backward()
                optimizer_G.step()
                scheduler_G.step()
            else:
                with torch.no_grad():
                    out, _ = model_G(images)
            
            if train_D and (not train_G or (A_loss < 0.4)):
                optimizer_D.zero_grad()
                fake_pred = model_D(out.detach())
                real_pred = model_D(images)
                fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
                real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
                loss_D = (fake_loss + real_loss) / 2
                loss_D.backward()
                optimizer_D.step()
                scheduler_D.step()
                D_loss = loss_D.item()
            
            c += 1
            looper.set_description(f"epoch {e}")
            if train_G and train_D:
                recon_loss_list.append(recon_loss.item())
                A_loss_list.append(A_loss)
                D_loss_list.append(D_loss)
                looper.set_postfix(loss_rec=recon_loss.item(),
                                   A_loss=A_loss, D_loss=D_loss,
                                   lr=optimizer_G.param_groups[0]['lr'])
            elif train_G:
                recon_loss_list.append(recon_loss.item())
                looper.set_postfix(loss_rec=recon_loss.item(),
                                   lr=optimizer_G.param_groups[0]['lr'])
            else:
                D_loss_list.append(D_loss)
                looper.set_postfix(D_loss=D_loss, lr=optimizer_D.param_groups[0]['lr'])
        
        if e % save_step == 0:
            if train_G:
                torch.save(model_G.state_dict(), f"models/{name}.pt")
                if images.shape[0] > 5:
                    images = images[:5]
                model_G.eval()
                with torch.no_grad():
                    out, _ = model_G(images)
                utils.save_image(
                    torch.cat([images, out], 0),
                    f"samples/{name}/epoch-{epoch_exist + e}.png",
                    nrow=images.shape[0],
                    normalize=True
                )
                model_G.train()
            if train_D:
                torch.save(model_D.state_dict(), f"models/{name}-D.pt")
    
    print(f"Train time: {time.time() - tt}")
    if loss_plot:
        if train_G:
            plt.plot(range(1, len(recon_loss_list)+1), recon_loss_list, label="recon_loss")
            plt.legend(loc="upper right")
        if train_G and train_D:
            plt.plot(range(1, len(A_loss_list)+1), A_loss_list, label="A_loss")
            plt.legend(loc="upper right")
        if train_D:
            plt.plot(range(1, len(D_loss_list)+1), D_loss_list, label="D_loss")
            plt.legend(loc="upper right")
        fig_exist = len(glob.glob(f"samples/{name}/Figure*"))
        plt.savefig(f"samples/{name}/Figure{fig_exist+1}.png")
        # plt.show()


def test(name, model):
    path = "test/*.png"
    device = "cuda"
    H, W = 9*64, 16*64
    dataset = AcDataset(path, (H, W), test=True)
    model.eval().to(device)
    load_model_file(model, f"{name}.pt")
    image_list = []
    for image in dataset:
        C, H, W = image.shape
        image_list.append(image.view(1, C, H, W))
    images = torch.cat(image_list, dim=0)
    test_exist = len(glob.glob(f"samples/{name}/test*"))
    with torch.no_grad():
        images = images.to(device)
        out, _ = model(images)
        utils.save_image(
            torch.cat([images, out], 0),
            f"samples/{name}/test{test_exist}.png",
            nrow=images.shape[0],
            normalize=True
        )


if __name__ == '__main__':
    name = "vqvae2"

    if not os.path.exists(f"samples/{name}"):
        os.mkdir(f"samples/{name}")
    
    G = VQVAE()
    D = Discriminator1()
    # G.summary()
    # D.summary()
    # train(name, G, D, 20, True, False, bs_G=8)
    # train(name, G, D, 10, False, True)
    train(name, G, D, 100, True, True, bs_G=6)
    # test(name, G)
