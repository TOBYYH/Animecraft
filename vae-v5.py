from vae import *


class _VAE(nn.Module):
    def __init__(self, act="ELU"):
        super().__init__()
        print("VAE initializing...")
        
        S = 128
        C = (S, S*2)
        C_Z = 8
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, C[0], 8, 4, 2),
            ResBlock2(C[0], act),
            nn.Conv2d(C[0], C[1], 4, 2, 1),
            ResBlock2(C[1], act),
            activition(act)
        )
        self.encoder_b = nn.Conv2d(C[1], C_Z*2, 1, 1, 0)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(C[1], C[1], 4, 2, 1),
            ResBlock2(C[1], act),
            nn.Conv2d(C[1], C[1], 4, 2, 1),
            ResBlock2(C[1], act),
            activition(act)
        )
        self.encoder_t = nn.Conv2d(C[1], C_Z*2, 1, 1, 0)
        
        self.decoder_t = nn.Sequential(
            nn.Conv2d(C_Z, C[1], 1, 1, 0),
            ResBlock2(C[1], act),
            nn.ConvTranspose2d(C[1], C[1], 4, 2, 1),
            ResBlock2(C[1], act),
            nn.ConvTranspose2d(C[1], C_Z, 4, 2, 1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(C_Z*2, C[1], 1, 1, 0),
            ResBlock2(C[1], act),
            nn.ConvTranspose2d(C[1], C[0], 4, 2, 1),
            ResBlock2(C[0], act),
            nn.ConvTranspose2d(C[0], 3, 8, 4, 2),
            activition("tanh")
        )
    
    def encode(self, x):
        z = self.encoder1(x)
        b = self.encoder_b(z)
        mean, logvar = torch.chunk(b, 2, dim=1)
        std = torch.exp(logvar * 0.5)
        latent_loss = torch.mean(torch.sum(mean**2 + torch.exp(logvar) - logvar - 1, dim=1) * 0.5)
        bottom = mean + torch.randn_like(mean) * std
        z = self.encoder2(z)
        t = self.encoder_t(z)
        mean, logvar = torch.chunk(t, 2, dim=1)
        std = torch.exp(logvar * 0.5)
        latent_loss += torch.mean(torch.sum(mean**2 + torch.exp(logvar) - logvar - 1, dim=1) * 0.5)
        top = mean + torch.randn_like(mean) * std
        return bottom, top, latent_loss
    
    def decode(self, b, t):
        z = self.decoder_t(t)
        z = torch.cat([z, b], dim=1)
        return self.decoder(z)
    
    def forward(self, x):
        b, t, latent_loss = self.encode(x)
        return self.decode(b, t), latent_loss

    def summary(self):
        H, W = 9*64, 16*64
        device, dtype = "cuda", torch.float32
        self.eval().to(device)
        torchinfo.summary(self, input_size=(1, 3, H, W), device=device)
        self.eval().to(device, dtype)
        z = torch.randn([1, 3, H, W], dtype=dtype, device=device)
        bottom, top, _ = self.encode(z)
        print(bottom.shape, top.shape)
        looper = tqdm(range(100))
        t = time.time()
        for _ in looper:
            z, _ = self(z.detach())
        t = time.time() - t
        print(f"{100. / t} /s")


if __name__ == '__main__':
    name = "AcVAE-v5"

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=-1)
    parser.add_argument("--epoch", type=int, default=10)
    opt = parser.parse_args()
    if not os.path.exists(f"samples/{name}"):
        os.mkdir(f"samples/{name}")
    
    G = _VAE()
    D = Discriminator1()
    if opt.mode == 1:
        train(name, G, D, opt.epoch, True, False)
    elif opt.mode == 2:
        train(name, G, D, opt.epoch, False, True)
    elif opt.mode == 3:
        train2(name, G, D, opt.epoch, True, True)
    else:
        # G.summary()
        # D.summary()
        test(name, G)
