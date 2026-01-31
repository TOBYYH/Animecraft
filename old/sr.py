from torch import nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

from utils import *
import models_sr, models_sr2
from models_lab1 import AutoEncoder


class SR(AcDatasetForSR):
    def __init__(self, path, W:int, H:int) -> None:
        super().__init__(path, W, H, W*4, H*4)
        self.device = "cuda"
        # self.model = AcSuperResolution()
        # load_model_file(self.model, "AcSuperResolution.pt")
        # self.AE = AutoEncoder()
        # load_model_file(self.AE, "AutoEncoder.pt")
        self.model = models_sr2.AcSuperResolution()
        load_model_file(self.model, "AcSuperResolution.pt")
        self.AE = AutoEncoder()
        load_model_file(self.AE, "AutoEncoder.pt")
    
    def train_AE(self, batch_size, epoch:int):
        self.enable_data_enhance()
        loader = DataLoader(self, batch_size=batch_size, shuffle=True, num_workers=6)
        optimizer = torch.optim.AdamW(self.AE.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5,
                                                        total_steps=epoch*(len(self)//batch_size+1),
                                                        div_factor=10.,
                                                        final_div_factor=2.)
        self.AE.train()
        self.AE.to(self.device)
        loss_func = nn.MSELoss()
        n = 0
        e = 1
        tt = time.time()
        for e in range(1, epoch + 1):
            looper = tqdm(loader, total=len(loader))
            for data in looper:
                hr = data[0].to(self.device)
                
                optimizer.zero_grad()
                out = self.AE(hr.detach().requires_grad_(True))
                loss = loss_func(out, hr)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                n += out.shape[0]
                looper.set_description(f"epoch: {e}")
                looper.set_postfix(loss=loss.item(), sample_num=n, lr=optimizer.param_groups[0]['lr'])
            
            torch.save(self.AE.state_dict(), f"models/AutoEncoder.pt")
        
        print(f"Train time: {time.time() - tt}")
    
    def test_AE(self):
        self.AE.eval()
        self.AE.to(self.device)
        hr = random.choice(self)[0]
        hr = hr.view(1, 3, hr.shape[1], hr.shape[2]).to(self.device)
        print(hr.shape)
        out = self.AE(hr)
        plt.subplot(1, 2, 1)
        plt.imshow(tensor2image_np(hr))
        plt.subplot(1, 2, 2)
        plt.imshow(tensor2image_np(out))
        plt.show()
        
        z = self.AE.encode(hr)
        plot_h, plot_w = 4, 8
        for h in range(plot_h):
            for w in range(plot_w):
                index = plot_w * h + w
                plt.subplot(plot_h, plot_w, index + 1)
                plt.imshow(z[0, index].cpu().detach().numpy())
        plt.show()
        
        z = torch.randn_like(z)
        plt.imshow(tensor2image_np(self.AE.decode(z)))
        plt.show()
    
    def train(self, epoch:int):
        self.enable_data_enhance()
        loader = DataLoader(self, batch_size=16, shuffle=True, num_workers=6)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-6)
        self.model.train()
        self.model.to(self.device)
        loss_func = nn.MSELoss()
        n = 0
        e = 1
        tt = time.time()
        for e in range(1, epoch + 1):
            looper = tqdm(loader, total=len(loader))
            for data in looper:
                hr, lr = data[0].to(self.device), data[1].to(self.device)
                
                optimizer.zero_grad()
                out = self.model(lr.detach())
                loss = loss_func(out, hr)
                loss.backward()
                optimizer.step()
                
                n += out.shape[0]
                looper.set_description(f"epoch: {e}")
                looper.set_postfix(loss=loss.item(), sample_num=n, lr=optimizer.param_groups[0]['lr'])
            
            torch.save(self.model.state_dict(), f"models/AcSuperResolution.pt")
        
        print(f"Train time: {time.time() - tt}")
    
    def test(self):
        self.model.eval()
        self.model.to(self.device)
        hr, lr = random.choice(self)
        print(hr.shape, lr.shape)
        plt.subplot(2, 2, 1)
        plt.imshow(tensor2image_np(lr))
        plt.subplot(2, 2, 2)
        plt.imshow(tensor2image_np(self.model(lr.view(1, 3, lr.shape[1], lr.shape[2]).to(self.device))))
        plt.subplot(2, 2, 4)
        plt.imshow(tensor2image_np(hr))
        plt.show()
    
    def sr(self, path):
        os.makedirs("sr", exist_ok=True)
        self.model.eval()
        self.model.to(self.device)
        paths = glob.glob(path)
        i = 0
        looper = tqdm(paths)
        for p in looper:
            image = Image.open(p)
            image = np.array(image, dtype=np.float32) / (255.0 / 2.0) - 1.0
            image = torch.tensor(image, dtype=torch.float32, device=self.device).permute(2, 0, 1)
            image = image.view(1, 3, image.shape[1], image.shape[2])
            plt.imsave(f"sr/{i}.jpg", tensor2image_np(self.model(image)))
            i += 1


if __name__ == '__main__':
    H, W = 104, 184
    path = "../LL/*"
    # path = "samples/v6/14/*"
    sr = SR(path, W, H)
    # sr.train_AE(16, 10)
    sr.test_AE()
    # sr.train(10)
    # sr.test()
    # sr.sr("samples/v7/60/*")
