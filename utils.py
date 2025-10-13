import os
import glob
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
import numpy as np
from PIL import Image,ImageOps
import random
import math
import matplotlib.pylab as plt
import torchvision
import torchvision.transforms as T
from torchvision.transforms import v2


def torch_init():
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")


def load_model_file(model, model_file):
    file = "models/" + model_file
    if os.path.isfile(file):
        print(f"Model file exists, load model file: {model_file}")
        model.load_state_dict(torch.load(file, map_location="cpu", weights_only=True))
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
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x:torch.Tensor):
        if self.inplace:
            return x.atan_()
        else:
            return x.atan()


class MyActivation1(nn.Module):
    def __init__(self):
        super().__init__()
    
    def __forward(self, x):
        return x / (x ** 2 * 0.25 + 1.)
    
    def forward(self, x):
        return checkpoint(self.__forward, x)


class FixedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = math.log(2, math.e) * 0.5
    
    def __forward(self, x):
        return F.softplus(x, beta=2).add_(-self.b)
    
    def forward(self, x):
        return checkpoint(self.__forward, x)
    
    def test(self):
        x = torch.linspace(-10, 10, 1000, dtype=torch.float32)
        y = self.__forward(x)
        plt.plot(x, y)
        plt.plot(x, torch.zeros_like(x))
        plt.show()


def activition(name, inplace=False) -> nn.Module:
    if name == "ReLU":
        return nn.ReLU(inplace=inplace)
    elif name == "LReLU":
        return nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "SiLU":
        return nn.SiLU(inplace=inplace)
    elif name == "ELU":
        return nn.ELU(inplace=inplace)
    elif name == "softplus":
        return nn.Softplus()
    elif name == "fixedsoftplus":
        return FixedSoftplus()
    elif name == "atan":
        return ArcTan(inplace)
    elif name == "my1":
        return MyActivation1()
    else:
        raise NotImplementedError(name)


class AcDataset(Dataset):
    def __init__(self, path, size, dtype=torch.float32, test=False) -> None:
        super().__init__()
        self.paths = glob.glob(path)
        ratio = size[1] / size[0]
        if test:
            self.transforms = v2.Compose([
                v2.Resize(size),
                v2.ToDtype(dtype, scale=True)
            ])
        else:
            self.transforms = v2.Compose([
                v2.RandomResizedCrop(size, (0.5, 1.0), (ratio*0.9, ratio*1.1)),
                v2.RandomHorizontalFlip(0.5),
                v2.ToDtype(dtype, scale=True)
            ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        # image = Image.open(self.paths[index]).convert("RGB")
        image = torchvision.io.read_image(self.paths[index], torchvision.io.ImageReadMode.RGB)
        return self.transforms(image) * 2 - 1
    
    def test(self, plot_h, plot_w):
        for h in range(plot_h):
            for w in range(plot_w):
                image = random.choice(self)
                plt.subplot(plot_h, plot_w, plot_w*h+w+1)
                plt.imshow(tensor2image_np(image))
        plt.show()


def tensor2image_np(image:torch.Tensor):
    assert image.shape[0] == 1 or image.dim() == 3
    H, W = image.shape[-2:]
    image = image.cpu().detach().view(3, H, W).permute(1, 2, 0).numpy()
    print(f"tensor2image_np: [{image.min()}, {image.max()}]")
    if image.min() < 0:
        image = image * 0.5 + 0.5
    return image.clip(0.0, 1.0)


class Conv(nn.Module):
    def __init__(self, c_in, c_out, k, s, p, act="tanh") -> None:
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p),
            group_norm(c_out),
            activition(act, True)
        )
    
    def forward(self, x):
        return self.nn(x)


class ConvT(nn.Module):
    def __init__(self, c_in, c_out, k, s, p, act="tanh") -> None:
        super().__init__()
        self.nn = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, k, s, p),
            group_norm(c_out),
            activition(act, True)
        )
    
    def forward(self, x):
        return self.nn(x)


class InceptionBlock(nn.Module):
    def __init__(self, c, k, g, t_emb_dim=0, act="tanh") -> None:
        super().__init__()
        assert c % 4 == 0
        self.norm = group_norm(c)
        p = (k - 1) // 2
        _c = c // 4
        self.conv3x3 = nn.Conv2d(_c, _c, 3, 1, 1)
        self.convw = nn.Conv2d(_c, _c, (1, k), 1, (0, p), groups=g)
        self.convh = nn.Conv2d(_c, _c, (k, 1), 1, (p, 0), groups=g)
        self.linear_emb = None
        if t_emb_dim != 0:
            self.linear_emb = nn.Linear(t_emb_dim, c)
        self.conv_out = nn.Sequential(
            activition(act),
            nn.Conv2d(c, c, 3, 1, 1)
        )
    
    def forward(self, x, t_emb=None):
        z = self.norm(x)
        z1, z2, z3, z4 = torch.split(z, z.shape[1]//4, dim=1)
        z = torch.cat([
            self.conv3x3(z1),
            self.convw(z2),
            self.convh(z3),
            z4
        ], dim=1)
        if self.linear_emb is not None:
            assert t_emb is not None
            z = z + self.linear_emb(t_emb)[:, :, None, None]
        z = self.conv_out(z)
        return x + z


class ResBlock(nn.Module):
    def __init__(self, c, k=3, g=1, t_emb_dim=0, act="tanh") -> None:
        super().__init__()
        self.label = 'r'
        assert k % 2 == 1
        p = (k - 1) // 2
        self.conv1 = nn.Sequential(
            group_norm(c),
            activition(act, True),
            nn.Conv2d(c, c, k, 1, p, groups=g)
        )
        if t_emb_dim != 0:
            self.linear_emb = nn.Linear(t_emb_dim, c)
        else:
            self.linear_emb = None
        self.conv2 = nn.Sequential(
            group_norm(c),
            activition(act, True),
            nn.Conv2d(c, c, 3, 1, 1)
        )
        self.conv3 = nn.Conv2d(c, c, 1, 1, 0)
    
    def forward(self, x, t_emb=None):
        z = self.conv1(x)
        if self.linear_emb is not None:
            assert t_emb is not None
            z = z + self.linear_emb(t_emb)[:, :, None, None]
        z = self.conv2(z)
        return z + self.conv3(x)


class SelfAttention(nn.Module):
    def __init__(self, c, num_heads, act="tanh") -> None:
        super().__init__()
        self.label = 's'
        k, s = 4, 4
        
        self.down = nn.Sequential(
            group_norm(c),
            nn.Conv2d(c, c, k, s, 0)
        )
        self.attention = nn.MultiheadAttention(c, num_heads, batch_first=True)
        self.up = nn.ConvTranspose2d(c, c, k, s, 0)
        # self.conv_in = nn.Conv2d(c, c, 1, 1, 0)
    
    def forward(self, x):
        z = self.down(x)
        N, C, H, W = z.shape
        z = z.view(N, C, H * W).transpose(1, 2)
        z, _ = self.attention(z, z, z)
        z = z.transpose(1, 2).view(N, C, H, W)
        return x + self.up(z)


class CrossAttention(nn.Module):
    def __init__(self, c, context_dim, num_heads, act="tanh") -> None:
        super().__init__()
        self.label = 'c'
        self.context_dim = context_dim

        self.down = nn.Sequential(
            group_norm(c),
            nn.Conv2d(c, c, 4, 4, 0)
        )
        self.context_linear = nn.Linear(context_dim, c)
        self.attention = nn.MultiheadAttention(c, num_heads, batch_first=True)
        self.up = nn.ConvTranspose2d(c, c, 4, 4, 0)
        self.conv_in = nn.Conv2d(c, c, 1, 1, 0)
    
    def forward(self, x, context=None):
        assert context is not None
        assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
        z = self.down(x)
        N, C, H, W = z.shape
        z = z.view(N, C, H * W).transpose(1, 2)
        con = self.context_linear(context)
        z, _ = self.attention(z, con, con)
        z = z.transpose(1, 2).view(N, C, H, W)
        z = self.up(z)
        return z + self.conv_in(x)


class Block0(nn.Module):
    def __init__(self, config_dict:dict, act="tanh") -> None:
        super().__init__()
        self.res_num = config_dict["res_num"]
        if "t_emb_dim" in config_dict.keys():
            self.t_emb_dim = config_dict["t_emb_dim"]
        else:
            self.t_emb_dim = 0
        if "self_attention" in config_dict.keys():
            assert len(config_dict["self_attention"]) == self.res_num
            self.sa = config_dict["self_attention"]
        else:
            self.sa = [0 for _ in range(self.res_num)]
        if "cross_attention" in config_dict.keys():
            assert len(config_dict["cross_attention"]) == self.res_num
            assert "context_dim" in config_dict.keys()
            context_dim = config_dict["context_dim"]
            self.ca = [(context_dim, x) for x in config_dict["cross_attention"]]
        else:
            self.ca = [(0, 0) for _ in range(self.res_num)]

        c = config_dict["c"]
        self.conv_in = Conv(c, c, 1, 1, 0, act)
        res_list = []
        for i in range(self.res_num):
            res_list.append(ResBlock(c//2, self.sa[i], self.ca[i], self.t_emb_dim, act))
        self.res_blocks = nn.ModuleList(res_list)
        c_out = c // 2 * (self.res_num + 2)
        self.conv_out = Conv(c_out, c, 1, 1, 0, act)
    
    def forward(self, x, t_emb=None, context=None):
        z = self.conv_in(x)
        z0, z = torch.split(z, z.shape[1]//2, dim=1)
        z_list = [z0, z]
        for i in range(self.res_num):
            z = self.res_blocks[i](z, t_emb, context)
            z_list.append(z)
        z = torch.cat(z_list, dim=1)
        return self.conv_out(z)


class Block(nn.Module):
    def __init__(self, config_dict:dict, out=None) -> None:
        """
        c, layers, kernel_size, group, activation, t_emb_dim, context_dim, num_heads
        layers: r->res_block, s->self_attention, c->cross_attention
        num_heads: (self_attention, cross_attention)

        out: ("down"/"up", c_out)
        """
        super().__init__()
        c = config_dict["c"]
        if "activation" in config_dict.keys():
            act = config_dict["activation"]
        else:
            act = "tanh"
        if "t_emb_dim" in config_dict.keys():
            self.t_emb_dim = config_dict["t_emb_dim"]
        else:
            self.t_emb_dim = 0
        if "num_heads" in config_dict.keys():
            num_heads = config_dict["num_heads"]
        else:
            num_heads = None
        if "kernel_size" in config_dict.keys():
            k = config_dict["kernel_size"]
        else:
            k = 3
        if "group" in config_dict.keys():
            g = config_dict["group"]
        else:
            g = 1
        
        layer_list = []
        for layer in config_dict["layers"]:
            if layer == 'r':
                layer_list.append(ResBlock(c, k, g, self.t_emb_dim, act))
            elif layer == 's':
                assert num_heads is not None
                layer_list.append(SelfAttention(c, num_heads[0], act))
            elif layer == 'c':
                assert num_heads is not None
                assert "context_dim" in config_dict.keys()
                layer_list.append(CrossAttention(c, config_dict["context_dim"], num_heads[1], act))
            else:
                raise NotImplementedError(layer)
        self.layers = nn.ModuleList(layer_list)

        if out is not None:
            if out[0] == "down":
                self.out = nn.Conv2d(c, out[1], 4, 2, 1)
            elif out[0] == "up":
                self.out = nn.ConvTranspose2d(c, out[1], 4, 2, 1)
            else:
                raise NotImplementedError(out)
        else:
            self.out = None
    
    def forward(self, x, t_emb=None, context=None):
        z = x
        for layer in self.layers:
            if layer.label == 'r':
                z = layer(z, t_emb)
            elif layer.label == 's':
                z = layer(z)
            elif layer.label == 'c':
                z = layer(z, context)
        if self.out is not None:
            z = self.out(z)
        return z


if __name__ == '__main__':
    # SR_dataset_gen()
    # test_AcDatasetForSR()
    # H, W = 104*4, 184*4
    # dataset = AcDataset("../LL/*", W, H)
    # dataset.enable_data_enhance()
    # dataset.test()
    
    # dataset = DatasetForFilter("frames")
    # dataset.test()
    dataset = AcDataset("frames/*.png", (9*64, 16*64))
    print(len(dataset))
    dataset.test(2, 3)
    
    # block = Block({
    #     "c": 32,
    #     "activation": "ELU",
    #     "layers": "rscrscrscrsc",
    #     "t_emb_dim": 64,
    #     "context_dim": 256,
    #     "num_heads": (4, 2)
    # })
    # x = torch.randn([3, 32, 128, 128], dtype=torch.float32, device="cpu")
    # t_emb = torch.randn([3, 64], dtype=torch.float32, device="cpu")
    # context = torch.randn([3, 16, 256], dtype=torch.float32, device="cpu")
    # print(block(x, t_emb, context).shape)
    
    # FixedSoftplus().test()
