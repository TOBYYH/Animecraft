from vae import *


class _VAE(nn.Module):
    def __init__(self, act="ELU"):
        super().__init__()
        print("VAE initializing...")
        
        S = 128
        C = (S, S*2)
        C_Z = 8
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, C[0], 8, 4, 2),
            ResBlock1(C[0], act),
            nn.Conv2d(C[0], C[1], 4, 2, 1),
            ResBlock1(C[1], act),
            ResBlock1(C[1], act),
            nn.Conv2d(C[1], C_Z*2, 1, 1, 0)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(C_Z, C[1], 1, 1, 0),
            ResBlock1(C[1], act),
            ResBlock1(C[1], act),
            nn.ConvTranspose2d(C[1], C[0], 4, 2, 1),
            ResBlock1(C[0], act),
            nn.ConvTranspose2d(C[0], 3, 8, 4, 2)
        )
    
    def encode(self, x):
        z = self.encoder(x)
        mean, logvar = torch.chunk(z, 2, dim=1)
        # std = torch.exp(logvar)
        # latent_loss = torch.mean(z**2)
        std = torch.exp(logvar * 0.5)
        latent_loss = torch.mean(torch.sum(mean**2 + torch.exp(logvar) - logvar - 1, dim=1) * 0.5)
        out = mean + torch.randn_like(mean) * std
        return out, latent_loss
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, latent_loss = self.encode(x)
        return self.decode(z), latent_loss

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
            z, _ = self(z.detach())
        t = time.time() - t
        print(f"{100. / t} /s")
        flops, params = profile(self, inputs=[z.detach()])
        print(f"FLOPs: {flops / 1000.**3}G, params: {params / 1000.**2}M")


if __name__ == '__main__':
    torch.set_num_threads(8)

    name = "AcVAE-v3"
    if not os.path.exists(f"samples/{name}"):
        os.mkdir(f"samples/{name}")
    
    G = _VAE()
    D = Discriminator1()
    # G.summary()
    # D.summary()
    # train(name, G, D, 10, True, False, bs_G=8)
    # train(name, G, D, 10, False, True)
    # train(name, G, D, 50, True, True, bs_G=8)
    test(name, G)
