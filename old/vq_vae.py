from torch import nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

from utils import *
from models_vae import VQVAE


def train():
    epoch = 1000
    H, W = 104, 184
    path = "../LL/*"
    device = "cuda"
    dataset = AcDataset(path, W, H)
    dataset.enable_data_enhance()
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=6)
    model = VQVAE()
    load_model_file(model, "AcVQVAE.pt")
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_func = nn.MSELoss()
    n = 0
    e = 1
    tt = time.time()
    for e in range(1, epoch + 1):
        looper = tqdm(loader, total=len(loader))
        for data in looper:
            image = data.to(device)
            
            optimizer.zero_grad()
            out, vq_loss = model(image.detach())
            loss = loss_func(out, image) + vq_loss
            loss.backward()
            optimizer.step()
            
            n += out.shape[0]
            looper.set_description(f"epoch: {e}")
            looper.set_postfix(loss=loss.item(), sample_num=n, lr=optimizer.param_groups[0]['lr'])
        
        torch.save(model.state_dict(), f"models/AcVQVAE.pt")
    
    print(f"Train time: {time.time() - tt}")


def test():
    H, W = 104, 184
    path = "../LL/*"
    device = "cuda"
    dataset = AcDataset(path, W, H)
    model = VQVAE()
    load_model_file(model, "AcVQVAE.pt")
    model.eval()
    model.plot_codebook()
    model.to(device)
    image = random.choice(dataset).view(1, 3, H, W).to(device)
    plt.subplot(1, 2, 1)
    plt.imshow(tensor2image_np(image))
    plt.subplot(1, 2, 2)
    plt.imshow(tensor2image_np(model(image)[0]))
    plt.show()


if __name__ == '__main__':
    # train()
    test()
