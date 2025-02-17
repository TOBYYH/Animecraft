from vqvae2 import *
import argparse


class _ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class _Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, mode):
        super().__init__()

        if mode == 'b':
            blocks = [
                nn.Conv2d(in_channel, channel // 4, 4, stride=2, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channel // 4, channel // 2, 4, stride=2, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif mode == 't':
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1)
            ]
        
        else:
            raise NotImplementedError(mode)

        for _ in range(n_res_block):
            blocks.append(_ResBlock(channel, n_res_channel))

        blocks.append(nn.LeakyReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class _Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, mode
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(_ResBlock(channel, n_res_channel))

        blocks.append(nn.LeakyReLU(inplace=True))

        if mode == 'b':
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.LeakyReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 2, channel // 4, 4, stride=2, padding=1),
                    nn.LeakyReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 4, out_channel, 4, stride=2, padding=1)
                ]
            )

        elif mode == 't':
            blocks.extend([
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1)
            ])

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class _VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=8,
        n_embed=512
    ):
        super().__init__()

        self.enc_b = _Encoder(in_channel, channel, n_res_block, n_res_channel, 'b')
        self.enc_t = _Encoder(channel, channel, n_res_block, n_res_channel, 't')
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = _Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, 't')
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        )
        self.dec = _Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            'b'
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
    name = "vqvae2-v0"

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
