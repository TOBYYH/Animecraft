import os
import glob
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from PIL import Image,ImageOps
import random
import math
import matplotlib.pylab as plt
# from nvidia.dali.pipeline import Pipeline


def torch_init():
    torch.set_num_threads(6)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")


def load_model_file(model, model_file):
    file = "models/" + model_file
    if os.path.isfile(file):
        print(f"Model file exists, load model file: {model_file}")
        model.load_state_dict(torch.load(file, map_location="cpu"))
        return True
    else:
        print(f"Model file doesn't exist, create new model: {model_file}")
        return False


def save_losses(losses, model_file):
    file = "models/" + model_file
    if os.path.isfile(file + ".npy"):
        np.save(file, np.append(np.load(file + ".npy"), losses))
    else:
        np.save(file, losses)


def group_norm(channel:int) -> nn.GroupNorm:
    if channel <= 8:
        return nn.GroupNorm(1, channel)
    elif channel <= 32:
        return nn.GroupNorm(4, channel)
    elif channel <= 128:
        return nn.GroupNorm(8, channel)
    elif channel <= 1024:
        return nn.GroupNorm(32, channel)
    else:
        return nn.GroupNorm(64, channel)


class ArcTan(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.atan_()


class MyActivation1(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x / (x ** 2 * 0.25 + 1.)


def activition(name, inplace=False):
    if name == "ReLU":
        return nn.ReLU(inplace)
    elif name == "SiLU":
        return nn.SiLU(inplace)
    elif name == "atan":
        return ArcTan()
    elif name == "my1":
        return MyActivation1()
    else:
        raise NotImplementedError(name)


class AcDataset(Dataset):
    def __init__(self, path, W=0, H=0) -> None:
        super().__init__()
        self.paths = glob.glob(path)
        self.shape = (W, H)
        
        self.dynamic_shape = False
        self.block_num = None
        self.block_size = None
        
        self.data_enhance = False
        self.rotation_max = 0.0
        self.pixel_drop_max = 0.0
    
    def enable_dynamic_shape(self, block_num, block_size):
        self.dynamic_shape = True
        self.block_num = block_num
        self.block_size = block_size
    
    def get_dynamic_shape(self, shape):
        aspect_ratio = float(shape[1]) / float(shape[0])
        t = math.sqrt(float(self.block_num) / aspect_ratio)
        h = int(t) * self.block_size
        w = int(t * aspect_ratio) * self.block_size
        return (h, w)
    
    def enable_data_enhance(self, rotation_max=10.0, pixel_drop_max=200):
        self.data_enhance = True
        self.rotation_max = rotation_max
        self.pixel_drop_max = pixel_drop_max
    
    def do_data_enhance(self, image:Image.Image):
        W, H = image.size
        d = (random.random() * 2.0 - 1.0) * self.rotation_max
        r = abs(d) * math.pi / 180.0
        k = (1.0 / math.tan(r)) + (1.0 / math.sin(r))
        t = k * k - 1.0
        l = max((k * W - H) / t, (k * H - W) / t)
        f = H / W
        offset_w = int(l / (f + math.tan(r))) + 1
        offset_h = int(f * offset_w) + 1
        image = image.rotate(d, resample=Image.Resampling.BILINEAR)
        image = image.crop((offset_w, offset_h, W - offset_w, H - offset_h))
        W, H = image.size
        pixel_drop_w = random.randint(0, self.pixel_drop_max)
        pixel_drop_h = H - int((W - pixel_drop_w) * f)
        if pixel_drop_h < 0:
            pixel_drop_h = 0
        W -= pixel_drop_w
        H -= pixel_drop_h
        offset_w = random.randint(0, pixel_drop_w)
        offset_h = random.randint(0, pixel_drop_h)
        image = image.crop((offset_w, offset_h, offset_w + W, offset_h + H))
        return image
    
    def get_image(self, index) -> Image.Image:
        image = Image.open(self.paths[index]).convert("RGB")
        if self.data_enhance:
            image = self.do_data_enhance(image)
        if self.dynamic_shape:
            return ImageOps.fit(image, self.get_dynamic_shape(image.size))
        if self.shape[0] * self.shape[1] == 0:
            return image
        return ImageOps.fit(image, self.shape)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        image = self.get_image(index)
        image = np.array(image, dtype=np.float32) / (255.0 / 2.0) - 1.0
        image = torch.tensor(image, dtype=torch.float32, device="cpu").permute(2, 0, 1)
        # print(image.shape)
        return image
    
    def test(self):
        image = random.choice(self)
        plt.imshow(tensor2image_np(image))
        plt.show()


class AcDatasetForSR(AcDataset):
    def __init__(self, path, LR_W, LR_H, HR_W, HR_H) -> None:
        super().__init__(path, 0, 0)
        self.shape_lr = (LR_W, LR_H)
        self.shape_hr = (HR_W, HR_H)
    
    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        if self.data_enhance:
            image = self.do_data_enhance(image)
        hr = ImageOps.fit(image, self.shape_hr)
        hr = np.array(hr, dtype=np.float32) / (255.0 / 2.0) - 1.0
        hr = torch.tensor(hr, dtype=torch.float32, device="cpu").permute(2, 0, 1)
        lr = ImageOps.fit(image, self.shape_lr)
        lr = np.array(lr, dtype=np.float32) / (255.0 / 2.0) - 1.0
        lr = torch.tensor(lr, dtype=torch.float32, device="cpu").permute(2, 0, 1)
        noise = torch.randn_like(lr)
        lr = lr + noise * 0.06
        return (hr, lr)


def tensor2image_np(image:torch.Tensor):
    assert image.shape[0] == 1 or image.dim() == 3
    H, W = image.shape[-2:]
    image = image.cpu().clone().detach().view(3, H, W).permute(1, 2, 0).numpy()
    image = image * 0.5 + 0.5
    return image.clip(0.0, 1.0)


def test_AcDatasetForSR():
    H, W = 104, 184
    dataset = AcDatasetForSR("/mnt/F89B887EA07147CB/DeepLearning/LL/*", W, H, W*4, H*4)
    dataset.enable_data_enhance()
    print(len(dataset))
    hr, lr = random.choice(dataset)
    print(hr.shape, lr.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(tensor2image_np(hr))
    plt.subplot(1, 2, 2)
    plt.imshow(tensor2image_np(lr))
    plt.show()


if __name__ == '__main__':
    # SR_dataset_gen()
    # test_AcDatasetForSR()
    H, W = 104*4, 184*4
    dataset = AcDataset("../LL/*", W, H)
    dataset.enable_data_enhance()
    dataset.test()
