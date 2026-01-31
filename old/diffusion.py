import os
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import argparse

from utils import *
import models_v0, models_v1, models_v2, models_v3
import models_v4, models_v5, models_v6, models_v7
from models_sr import AcSuperResolution


torch.set_num_threads(6)
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")


def position_encoding(t, dim:int):
    assert dim % 2 == 0
    k = np.arange(1, dim//2+1, 1, dtype=np.float32)
    w = 1.0 / (10000 ** (k * 2.0 / dim))
    pos_sin = np.sin(w * t).reshape(w.shape[0], 1)
    pos_cos = np.cos(w * t).reshape(w.shape[0], 1)
    return np.concatenate((pos_sin, pos_cos), axis=1).flatten()


class Diffusion:
    def __init__(self, version:int, path, W:int, H:int,
                 diff_pars, step_num:int, time_dim:int) -> None:
        self.device = "cuda"
        self.dataset = AcDataset(path, W, H)
        
        self.diff_pars = diff_pars
        self.step_num = step_num
        self.time_dim = time_dim
        self.step_range = None
        self.betas = self.__get_betas()
        self.alphas_p = self.__get_alphas_p()
        
        if version == -1:
            self.model = models_v7.AcDiffusionUNet(time_dim)
            self.model_files = ["AcDiffusionV7A.pt", "AcDiffusionV7B.pt"]
            self.load_steps = [500]
        elif version == 0:
            self.model = models_v0.AcDiffusionUNet(time_dim)
            self.model_files = ["AcDiffusionV0A.pt", "AcDiffusionV0B.pt"]
            self.load_steps = [500]
        elif version == 1:
            self.model = models_v1.AcDiffusionUNet(time_dim)
            self.model_files = ["AcDiffusionV1A.pt", "AcDiffusionV1B.pt"]
            self.load_steps = [500]
        elif version == 2:
            self.model = models_v2.AcDiffusionUNet(time_dim)
            self.model_files = ["AcDiffusionV2A.pt", "AcDiffusionV2B.pt"]
            self.load_steps = [500]
        elif version == 3:
            self.model = models_v3.AcDiffusionUNet(time_dim)
            self.model_files = ["AcDiffusionV3A.pt", "AcDiffusionV3B.pt"]
            self.load_steps = [500]
        elif version == 4:
            self.model = models_v4.AcDiffusionUNet(time_dim)
            self.model_files = ["AcDiffusionV4A.pt", "AcDiffusionV4B.pt"]
            self.load_steps = [500]
        elif version == 5:
            self.model = models_v5.AcDiffusionUNet(time_dim)
            self.model_files = ["AcDiffusionV5A.pt", "AcDiffusionV5B.pt"]
            self.load_steps = [500]
        elif version == 6:
            self.model = models_v6.AcDiffusionUNet(time_dim)
            self.model_files = ["AcDiffusionV6A.pt", "AcDiffusionV6B.pt"]
            self.load_steps = [500]
        elif version == 7:
            self.model = models_v7.AcDiffusionUNet(time_dim)
            self.model_files = ["AcDiffusionV7A.pt", "AcDiffusionV7B.pt"]
            self.load_steps = [500]
        else:
            raise NotImplementedError
        # self.model = torch.compile(self.model)
        self.version = version
        assert len(self.model_files) == len(self.load_steps) + 1
    
    def __get_betas(self):
        r, start, end = self.diff_pars
        steps = np.linspace(0, self.step_num - 1, self.step_num, dtype=np.float32)
        f = 1 / (float(self.step_num) - steps * r)
        scale = (end - start) / (f[-1] - f[0])
        k = start - f[0] * scale
        return f * scale + k
    
    def __get_alphas_p(self):
        alphas_p = np.zeros_like(self.betas, dtype=np.float32)
        alphas_p[0] = 1 - self.betas[0]
        for i in range(1, alphas_p.size):
            alphas_p[i] = alphas_p[i - 1] * (1 - self.betas[i])
        return alphas_p
    
    def train(self, model_index, epoch):
        if len(self.load_steps) == 0:
            self.step_range = (10, self.step_num)
        else:
            if model_index == 0:
                self.step_range = (1, self.load_steps[0])
            elif model_index == len(self.load_steps):
                self.step_range = (self.load_steps[model_index - 1], self.step_num)
            else:
                assert 0 < model_index and model_index < len(self.load_steps)
                self.step_range = (self.load_steps[model_index - 1], self.load_steps[model_index])
        print(f"step_range: {self.step_range}, epoch: {epoch}, dataset_len: {len(self.dataset)}")
        self.dataset.enable_data_enhance()
        loader = DataLoader(self.dataset, batch_size=8, shuffle=True, num_workers=6)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5,
                                                        total_steps=epoch*(len(self.dataset)//8+1),
                                                        div_factor=20.,
                                                        final_div_factor=1000.)
        load_model_file(self.model, self.model_files[model_index])
        self.model.train()
        self.model.to(self.device)
        loss_func = nn.MSELoss()
        n = 0
        e = 1
        loss_list = []
        tt = time.time()
        for e in range(1, epoch + 1):
            looper = tqdm(loader, total=len(loader))
            for data in looper:
                image = data.to(self.device)
                noise = torch.randn_like(image)
                t = random.randint(self.step_range[0]-1, self.step_range[1]-1)
                image = image * np.sqrt(self.alphas_p[t]) + noise * np.sqrt(1. - self.alphas_p[t])
                pos_emb = position_encoding(t + 1, self.time_dim)
                pos_emb = torch.tensor(pos_emb, dtype=torch.float32, device=self.device).view(1, -1)
                
                optimizer.zero_grad()
                out = self.model(image.detach(), pos_emb)
                loss = loss_func(out, noise)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                n += out.shape[0]
                looper.set_description(f"epoch: {e}")
                looper.set_postfix(loss=loss.item(), sample_num=n, lr=optimizer.param_groups[0]['lr'])
                loss_np = np.array([0], dtype=np.float32)
                loss_np[0] = loss.data
                loss_list.append(loss_np)
        
        print(f"Train time: {time.time() - tt}")
        torch.save(self.model.state_dict(), f"models/{self.model_files[model_index]}")
        losses = np.array(loss_list, dtype=np.float32).flatten()
        save_losses(losses, self.model_files[model_index])
    
    def show_losses(self, model_index):
        losses = np.load(f"models/{self.model_files[model_index]}.npy")
        print(len(losses))
        steps = np.linspace(1, len(losses), len(losses), dtype=np.int32)
        plt.plot(steps, losses)
        plt.show()
    
    def denoising(self, image, step_start, step_end, save_path="", save_steps=False):
        assert self.step_num >= step_start and step_start > step_end and step_end > 0
        
        offset = 0
        for i in range(len(self.load_steps)):
            assert self.load_steps[i] > offset
            if step_start > offset and step_start <= self.load_steps[i]:
                self.model.load_state_dict(
                    torch.load(f"models/{self.model_files[i]}", map_location="cpu")
                )
                model_index = i
            offset = self.load_steps[i]
        if step_start > offset:
            self.model.load_state_dict(
                torch.load(f"models/{self.model_files[-1]}", map_location="cpu")
            )
            model_index = len(self.load_steps)
        
        self.model.eval()
        self.model.to(self.device)
        looper = tqdm(range(step_start, step_end - 1, -1))
        for t in looper:
            if model_index > 0:
                if t == self.load_steps[model_index - 1]:
                    model_index -= 1
                    self.model.load_state_dict(
                        torch.load(f"models/{self.model_files[model_index]}", map_location="cpu")
                    )
            tt = t - 1
            looper.set_postfix(step=tt, model_index=model_index)
            t_emb = position_encoding(t, self.time_dim)
            t_emb = torch.tensor(t_emb, dtype=torch.float32, device=self.device).view(1, -1)
            k1 = 1.0 / np.sqrt(1.0 - self.betas[tt])
            k2 = self.betas[tt] / np.sqrt(1. - self.alphas_p[tt])
            image = (image - self.model.cp(image.detach(), t_emb) * k2) * k1
            if tt > 0:
                noise = torch.randn(image.shape, dtype=torch.float32, device=self.device)
                v = (1.0 - self.alphas_p[tt - 1]) / (1.0 - self.alphas_p[tt]) * self.betas[tt]
                image = image + noise * np.sqrt(v)
            if save_steps:
                plt.imsave(save_path + f"/step-{tt}.jpg", tensor2image_np(image))
        
        return image
    
    def sample(self, sample_num, label):
        save_dir = f"samples/v{self.version}"
        os.makedirs(save_dir, exist_ok=True)
        exist_num = len(glob.glob(f"{save_dir}/*"))
        save_dir = f"{save_dir}/{exist_num + 1}"
        os.makedirs(save_dir)
        W, H = self.dataset.shape
        for i in range(1, sample_num + 1):
            z = torch.randn([1, 3, H, W], dtype=torch.float32, device=self.device)
            image = self.denoising(z, self.step_num, 1)
            plt.imsave(f"{save_dir}/{label}-sample{i}.jpg", tensor2image_np(image))
    
    def test(self, t):
        W, H = self.dataset.shape
        image = random.choice(self.dataset).to(self.device).view(1, 3, H, W)
        plt.subplot(2, 2, 1)
        plt.imshow(tensor2image_np(image))
        noise = torch.randn_like(image)
        image = image * np.sqrt(self.alphas_p[t]) + noise * np.sqrt(1. - self.alphas_p[t])
        image = self.denoising(image, t+1, 1)
        plt.subplot(2, 2, 2)
        plt.imshow(tensor2image_np(image))
        sr = AcSuperResolution()
        sr.eval()
        sr.to(self.device)
        load_model_file(sr, "AcSuperResolution.pt")
        plt.subplot(2, 2, 4)
        plt.imshow(tensor2image_np(sr(image)))
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=-1)
    parser.add_argument("--n", type=int, default=1)
    opt = parser.parse_args()
    print(f"version: {opt.version}, n: {opt.n}")
    version = opt.version
    H, W = 104, 184
    step_num = 1000
    time_dim = 1024
    diff_pars = (0.9, 0.0001, 0.04)
    path = "../LL/*"
    # path = "samples/dataset/*"
    diffusion = Diffusion(version, path, W, H, diff_pars, step_num, time_dim)
    # for i in range(opt.n):
    #     # diffusion.train(0, 10)
    #     # diffusion.train(1, 10)
    #     diffusion.sample(10, i+1)
    # diffusion.show_losses(0)
    diffusion.test(300)
