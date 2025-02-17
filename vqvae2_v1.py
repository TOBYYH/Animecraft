from vqvae2 import *
import argparse


class _ResBlock(nn.Module):
    def __init__(self, c, act="ELU"):
        super().__init__()

        self.conv = nn.Sequential(
            group_norm(c),
            nn.Conv2d(c, c, 4, 2, 1),
            activition(act, True),
            nn.ConvTranspose2d(c, c, 4, 2, 1)
        )

    def forward(self, x):
        return x + self.conv(x)


def _res_blocks(n_res, c, act="ELU"):
    blocks = []
    for _ in range(n_res):
        blocks.append(_ResBlock(c, act))
    return blocks


class _VQVAE(nn.Module):
    def __init__(
        self,
        channels=(128, 256),
        embed_dim=8,
        n_embed=1024,
        act="ELU"
    ):
        super().__init__()

        _c = channels[1]
        self.enc_b = nn.Sequential(
            nn.Conv2d(3, channels[0], 8, 4, 2),
            _ResBlock(channels[0], act),
            nn.Conv2d(channels[0], channels[1], 4, 2, 1),
            _ResBlock(channels[1], act),
            activition(act)
        )
        self.enc_t = nn.Sequential(
            nn.Conv2d(_c, _c, 4, 2, 1),
            _ResBlock(_c, act),
            nn.Conv2d(_c, _c, 4, 2, 1),
            _ResBlock(_c, act),
            activition(act)
        )

        self.quantize_conv_t = nn.Conv2d(_c, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)

        self.dec_t = nn.Sequential(
            nn.Conv2d(embed_dim, _c, 1),
            _ResBlock(_c, act),
            nn.ConvTranspose2d(_c, _c, 4, 2, 1),
            _ResBlock(_c, act),
            nn.ConvTranspose2d(_c, embed_dim, 4, 2, 1),
            activition(act)
        )

        self.quantize_conv_b = nn.Conv2d(_c+embed_dim, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, 2, 1),
            activition(act, True),
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, 2, 1)
        )

        self.dec = nn.Sequential(
            nn.Conv2d(embed_dim*2, _c, 1),
            _ResBlock(_c, act),
            nn.ConvTranspose2d(_c, channels[0], 4, 2, 1),
            _ResBlock(channels[0], act),
            nn.ConvTranspose2d(channels[0], 3, 8, 4, 2),
            nn.Tanh()
        )

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


if __name__ == '__main__':
    name = "vqvae2-v1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=-1)
    opt = parser.parse_args()
    if not os.path.exists(f"samples/{name}"):
        os.mkdir(f"samples/{name}")
    
    G = _VQVAE()
    D = Discriminator1()
    if opt.mode == 1:
        train(name, G, D, 50, True, False, bs_G=8)
    elif opt.mode == 2:
        train(name, G, D, 10, False, True)
    elif opt.mode == 3:
        train(name, G, D, 100, True, True, bs_G=8)
    else:
        G.summary()
        # D.summary()
        # test(name, G)
