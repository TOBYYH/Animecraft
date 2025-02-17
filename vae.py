from utils import *
from torch.utils.data import DataLoader
from torchvision import utils
import torchinfo
import time
import argparse
from tqdm import tqdm


class ResBlock0(nn.Module):
    def __init__(self, c, act="tanh") -> None:
        super().__init__()
        self.nn = nn.Sequential(
            Conv(c, c, 3, 1, 1, act),
            Conv(c, c, 3, 1, 1, act),
        )
    
    def forward(self, x):
        return x + self.nn(x)


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


class ResBlock2(nn.Module):
    def __init__(self, c, act="tanh") -> None:
        super().__init__()
        self.nn = nn.Sequential(
            group_norm(c),
            nn.Conv2d(c, c, 4, 2, 1),
            activition(act, True),
            nn.ConvTranspose2d(c, c, 4, 2, 1)
        )
    
    def forward(self, x):
        return x + self.nn(x)


class Discriminator0(nn.Module):
    def __init__(self, act="ELU"):
        super().__init__()
        print("Discriminator initializing...")
        
        C_IN = 32
        S = 64
        C = (S, S*2, S*4)
        self.layers = nn.Sequential(
            Conv(3, C_IN, 4, 2, 1, act),
            Conv(C_IN, C[0], 4, 2, 1, act),
            ResBlock0(C[0], act),
            Conv(C[0], C[1], 4, 2, 1, act),
            ResBlock0(C[1], act),
            Conv(C[1], C[2], 4, 2, 1, act),
            ResBlock0(C[2], act),
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


class Discriminator(nn.Module):
    def __init__(self, act="ELU"):
        super().__init__()
        print("Discriminator initializing...")
        
        C_IN = 32
        S = 64
        C = (S, S*2, S*4)
        block1 = {
            "c": C[0],
            "activation": act,
             "layers": "rr"
        }
        block2 = {
            "c": C[1],
            "activation": act,
             "layers": "rrr"
        }
        block3 = {
            "c": C[2],
            "activation": act,
             "layers": "rrrr"
        }
        self.layers = nn.Sequential(
            nn.Conv2d(3, C_IN, 4, 2, 1),
            activition(act, True),
            nn.Conv2d(C_IN, C[0], 4, 2, 1),
            Block(block1, ("down", C[1])),
            Block(block2, ("down", C[2])),
            Block(block3),
            activition(act, True),
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


class VAE(nn.Module):
    def __init__(self, act="ELU"):
        super().__init__()
        print("VAE initializing...")
        
        C_IN = 32
        S = 128
        C = (S, S*2)
        C_Z = 8
        block1 = {
            "c": C[0],
            "activation": act,
            "layers": "rrrr",
        }
        block2 = {
            "c": C[1],
            "activation": act,
            "layers": "rrsrrsrr",
            "num_heads": (4,)
        }
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, C_IN, 4, 2, 1),
            activition(act, True),
            nn.Conv2d(C_IN, C[0], 4, 2, 1),
            Block(block1, ("down", C[1])),
            Block(block2),
            Conv(C[1], C_Z*2, 1, 1, 0, act)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(C_Z, C[1], 1, 1, 0),
            Block(block2, ("up", C[0])),
            Block(block1, ("up", C_IN)),
            activition(act, True),
            nn.ConvTranspose2d(C_IN, 3, 4, 2, 1)
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
        # self.eval().to(device)
        # torchinfo.summary(self, input_size=(1, 3, H, W), device=device)
        self.eval().to(device, dtype)
        z = torch.randn([1, 3, H, W], dtype=dtype, device=device)
        looper = tqdm(range(100))
        t = time.time()
        for _ in looper:
            z, _ = self(z.detach())
        t = time.time() - t
        print(f"{100. / t} /s")


def train(name, model_G, model_D, epoch, train_G=True, train_D=False,
          loss_plot=True, bs_G=8, bs_D=16):
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
        scheduler_G = torch.optim.lr_scheduler.OneCycleLR(optimizer_G, max_lr=1e-5,
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
        scheduler_D = torch.optim.lr_scheduler.OneCycleLR(optimizer_D, max_lr=1e-5,
                                                          total_steps=epoch*len(loader),
                                                          div_factor=10.0,
                                                          final_div_factor=1.0)
    recon_loss_list = []
    A_loss_list = []
    D_loss_list = []
    # vae_weight = 1e-4
    gan_weight = 0.01
    A_loss = 0.0
    D_loss = 0.0
    c = 0
    tt = time.time()
    for e in range(1, epoch + 1):
        vae_weight = 0.01 / (math.exp(float(epoch_exist+e-10)) + 1.0) + 1e-5
        looper = tqdm(loader, total=len(loader))
        for data in looper:
            images = data.to(device).requires_grad_(False)
            
            if train_G:
                optimizer_G.zero_grad()
                out, latent_loss = model_G(images)
                recon_loss = F.mse_loss(out, images)
                loss = recon_loss + latent_loss * vae_weight
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


def train2(name, model_G, model_D, epoch, train_G=True, train_D=False,
           loss_plot=True, bs_G=8, bs_D=16):
    assert train_G or train_D
    save_step = 10
    if train_G:
        batch_size = bs_G
        turn_G, turn_D = True, False
    else:
        batch_size = bs_D
        turn_G, turn_D = False, True
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
        scheduler_G = torch.optim.lr_scheduler.OneCycleLR(optimizer_G, max_lr=1e-5,
                                                          total_steps=epoch*len(loader),
                                                          div_factor=10.0,
                                                          final_div_factor=0.5)
    else:
        model_G.eval().to(device, dtype)
    load_model_file(model_G, f"{name}.pt")
    
    if train_D:
        model_D.train().to(device, dtype)
        load_model_file(model_D, f"{name}-D.pt")
        optimizer_D = torch.optim.AdamW(model_D.parameters(), lr=1e-5)
        scheduler_D = torch.optim.lr_scheduler.OneCycleLR(optimizer_D, max_lr=1e-5,
                                                          total_steps=epoch*len(loader),
                                                          div_factor=10.0,
                                                          final_div_factor=0.5)
    recon_loss_list = []
    A_loss_list = []
    D_loss_list = []
    # vae_weight = 1e-4
    gan_weight = 0.01
    A_loss = 0.0
    D_loss = 0.0
    A_loss_m = 0.0
    D_loss_m = 0.0
    c = 0
    tt = time.time()
    for e in range(1, epoch + 1):
        vae_weight = 0.01 / (math.exp(float(epoch_exist+e-10)) + 1.0) + 1e-5
        looper = tqdm(loader, total=len(loader))
        for data in looper:
            images = data.to(device).requires_grad_(False)
            
            if train_G:
                optimizer_G.zero_grad()
                out, latent_loss = model_G(images)
                recon_loss = F.mse_loss(out, images)
                loss = recon_loss + latent_loss * vae_weight
                if train_D and D_loss_m < 0.5:
                    fake_pred = model_D(out)
                    loss_A = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
                    A_loss = loss_A.item()
                    loss += loss_A * gan_weight
                    A_loss_m = A_loss_m * 0.9 + A_loss * 0.1
                    # if A_loss_m < 0.1:
                    #     turn_G, turn_D = False, True
                    #     A_loss_m = 1.0
                loss.backward()
                optimizer_G.step()
            else:
                with torch.no_grad():
                    out, _ = model_G(images)
            
            if train_D and (not train_G or A_loss_m < 0.5):
                optimizer_D.zero_grad()
                fake_pred = model_D(out.detach())
                real_pred = model_D(images)
                fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
                real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
                loss_D = (fake_loss + real_loss) / 2
                loss_D.backward()
                optimizer_D.step()
                D_loss = loss_D.item()
                D_loss_m = D_loss_m * 0.9 + D_loss * 0.1
                # if train_G and D_loss_m < 0.05:
                #     turn_G, turn_D = True, False
                #     D_loss_m = 1.0
            
            c += 1
            looper.set_description(f"epoch {e}")
            if train_G and train_D:
                scheduler_G.step()
                scheduler_D.step()
                recon_loss_list.append(recon_loss.item())
                A_loss_list.append(A_loss)
                D_loss_list.append(D_loss)
                looper.set_postfix(rec_l=recon_loss.item(),
                                   A_l=A_loss, A_l_m=A_loss_m, D_l=D_loss, D_l_m=D_loss_m,
                                   lr=optimizer_G.param_groups[0]['lr'])
            elif train_G:
                scheduler_G.step()
                recon_loss_list.append(recon_loss.item())
                looper.set_postfix(loss_rec=recon_loss.item(),
                                   lr=optimizer_G.param_groups[0]['lr'])
            else:
                scheduler_D.step()
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
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    
    parser = argparse.ArgumentParser()
    VAE("ELU").summary()
    # Discriminator().summary()
    # train(20, True, True)
    # test()
