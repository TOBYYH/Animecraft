import glob
import random
import matplotlib.pylab as plt
import torch
from torch import nn
import numpy as np
from PIL import Image,ImageOps
import torchsummary
import math
import time

from models import AcSuperResolution, AcDiscriminator, AcDiffusionUNet
# from models_lab1 import AcDiffusionUNetLab
from utils import (
    AcDatasetForSR, AcDataset,
    tensor2image_np
)

torch.set_num_threads(6)


def get_betas(step_num, r, start, end):
    steps = np.linspace(0, step_num - 1, step_num, dtype=np.float32)
    f = 1 / (float(step_num) - steps * r)
    scale = (end - start) / (f[-1] - f[0])
    k = start - f[0] * scale
    return f * scale + k


def get_alphas_p(step_num, r, start, end, betas=None):
    if betas is None:
        betas = get_betas(step_num, r, start, end)
    alphas_p = np.zeros_like(betas, dtype=np.float32)
    alphas_p[0] = 1 - betas[0]
    for i in range(1, alphas_p.size):
        alphas_p[i] = alphas_p[i - 1] * (1 - betas[i])
    return alphas_p


def test_SR():
    H, W = 104, 184
    model_path="models/AcSuperResolution.pt"
    device = "cuda"
    dataset = AcDatasetForSR("SR_dataset")
    model = AcSuperResolution()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    hr, lr = random.choice(dataset)
    lr = lr.view(1, 3, H, W)
    sr = model(lr.to(device))
    lr = nn.functional.interpolate(lr, size=(H*4, W*4), mode='bicubic').clamp(min=-1.0, max=1.0)
    plt.subplot(2, 2, 1)
    plt.imshow(tensor2image_np(lr))
    plt.subplot(2, 2, 2)
    plt.imshow(tensor2image_np(sr))
    plt.subplot(2, 2, 4)
    plt.imshow(hr.permute(1, 2, 0).numpy() * 0.5 + 0.5)
    plt.show()


def test_SR_sample():
    H, W = 104, 184
    model_path="models/AcSuperResolution.pt"
    device = "cuda"
    dataset = AcDataset("samples/*")
    model = AcSuperResolution()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    lr = random.choice(dataset)
    lr = lr.view(1, 3, H, W)
    sr = model(lr.to(device))
    lr = nn.functional.interpolate(lr, size=(H*4, W*4), mode='bicubic').clamp(min=-1.0, max=1.0)
    plt.subplot(1, 2, 1)
    plt.imshow(tensor2image_np(lr))
    plt.subplot(1, 2, 2)
    plt.imshow(tensor2image_np(sr))
    plt.show()


def test_discriminator():
    times = 100
    path_SR="models/AcSuperResolution.pt"
    # path_SR="models/super_resolution.pt.epoch10"
    # model_path="models/AcDiscriminator.pt"
    model_path="models/epoch100-AcDiscriminator.pt"
    path = "d:/DeepLearning/waifu_dataset/*"
    # path = "/mnt/DF2E413D652A1785/DeepLearning/wife_test/*"
    # path = "d:/DeepLearning/waifu_test/*"
    device = "cuda"
    dataset = AcDatasetForSR(path, 512, 512, 128, 128)
    model = AcDiscriminator()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    model_SR = AcSuperResolution()
    model_SR.load_state_dict(torch.load(path_SR))
    model_SR.eval()
    model_SR.to(device)
    correct = 0
    for _ in range(times):
        hr, lr = random.choice(dataset)
        lr = lr.to(device).view(1, 3, 128, 128)
        hr = hr.to(device).view(1, 3, 512, 512)
        sr = model_SR(lr)
        
        out1 = model(sr)
        if out1[0, 0].data < 0.5:
            correct += 1
        out2 = model(hr)
        if out2[0, 0].data >= 0.5:
            correct += 1
        
        print(f"SR model output: {out1[0, 0].data}, HR model output: {out2[0, 0].data}")
    
    print(f"Correct: {correct}/{times * 2}, accuracy: {correct / (times * 2) * 100}%")


def summary_SR():
    device = "cpu"
    model = AcSuperResolution()
    model.to(device)
    torchsummary.summary(model, input_size=(3, 104, 184), device=device)


def summary_D():
    H, W = 448, 768
    device = "cpu"
    model = AcDiscriminator(H, W)
    model.to(device)
    torchsummary.summary(model, input_size=(3, H, W), device=device)


def summary_U():
    H, W = 104, 184
    step_num = 1000
    time_dim = 1024
    diff_pars = (0.95, 0.0001, 0.05)
    device = "cuda"
    model = AcDiffusionUNet(1024)
    model.eval()
    model.to(device)
    torchsummary.summary(model, input_size=(3, H, W), device=device)
    betas = get_betas(step_num, diff_pars[0], diff_pars[1], diff_pars[2])
    alphas_p = get_alphas_p(step_num, diff_pars[0], diff_pars[1], diff_pars[2], betas)
    z = torch.randn([1, 3, H, W], dtype=torch.float32, device=device)
    t = time.time()
    z = generation(model, z, step_num, time_dim, betas, alphas_p, 100, 1)
    print(time.time() - t)


def test2():
    x = torch.ones([2, 3, 5, 5], dtype=torch.float32, device="cpu")
    conv_t = nn.ConvTranspose2d(3, 2, (4, 4), (2, 2), (1, 1))
    x = conv_t(x)
    print(x.size())
    conv1 = nn.Conv2d(2, 3, (4, 4), (2, 2), (1, 1))
    x = conv1(x)
    print(x.size())
    print(x)
    x = torch.cat((x, x), 1)
    print(x)
    print(x.size())
    x = torch.ones([2, 3, 512, 512], dtype=torch.float32, device="cpu")
    conv = nn.Conv2d(3, 16, (8, 8), (2, 2), (3, 3))
    x = conv(x)
    print(x.size())
    x = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32, device="cpu", requires_grad=True)
    print(x ** 2 + 1)
    x = torch.ones([2, 3, 5, 5], dtype=torch.float32, device="cpu")
    conv = nn.Conv2d(3, 3, 3, 1, 1, groups=3)
    print(conv(x).shape)
    print(torch.cat((x, x), dim=0).shape)
    x = torch.zeros([5], dtype=torch.float32, device="cuda")
    print(x)
    x[1] = 1.
    print(x)
    for i in range(10, 0, -1):
        print(i)
    x = torch.ones([2, 3, 5, 5], dtype=torch.float32, device="cpu")
    conv_t = nn.ConvTranspose2d(3, 2, 2, 2, 0)
    x = conv_t(x)
    print(x.size())
    x = torch.ones([2, 16, 5, 5], dtype=torch.float32, device="cpu")
    conv = nn.Conv2d(16, 4, 3, 1, 1, groups=4)
    print(conv(x).shape)


def test3():
    H = 4
    W = 8
    step_num = 1000
    dt = torch.float32
    dev = "cpu"
    paths = glob.glob("/mnt/DF2E413D652A1785/DeepLearning/waifu_dataset/*")
    image = Image.open(random.choice(paths)).convert("RGB")
    image = ImageOps.fit(image, (512, 512))
    image = np.array(image, dtype=np.float32) / 255.
    print(image.transpose(2, 0, 1).shape)
    # image = torch.tensor(image, dtype=dt, device=dev)
    
    alphas_p = get_alphas_p(step_num, 0.96, 0.0001, 0.04)
    # print(alphas_p)
    c = 0
    for h in range(H):
        for w in range(W):
            t = step_num // (H * W) * c
            print(t)
            temp = image * np.sqrt(alphas_p[t]) + np.random.randn(*image.shape) * np.sqrt(1. - alphas_p[t])
            # noise = torch.randn(image.shape, dtype=dt, device=dev)
            # temp = image * np.sqrt(alphas_p[t]) + noise * np.sqrt(1. - alphas_p[t])
            plt.subplot(H, W, c + 1)
            plt.imshow(temp.clip(0.0, 1.0))
            c += 1
    
    plt.show()


def test_AcDataSetForDiff():
    H, W = 104, 184
    step_num = 1000
    time_dim = 1024
    diff_pars = (0.95, 0.0001, 0.05)
    path = "/mnt/DF2E413D652A1785/DeepLearning/LL/*"
    # path = "d:/DeepLearning/LL/*"
    data = AcDatasetForDiff(path, W, H, diff_pars, step_num, time_dim, 20, 100)
    data.enable_data_enhance()
    print(len(data))
    image, noise, t_emb = random.choice(data)
    print(image.shape, noise.shape, t_emb.shape)
    plt.subplot(1, 3, 1)
    plt.imshow(tensor2image_np(image))
    plt.subplot(1, 3, 2)
    plt.imshow(tensor2image_np(noise))
    plt.subplot(1, 3, 3)
    plt.imshow(t_emb.view(32, 32))
    plt.show()


def sample():
    sample_num = 50
    label = 'a'
    save_path = "samples"
    H, W = 104, 184
    step_num = 1000
    save_steps = False
    time_dim = 1024
    diff_pars = (0.95, 0.0001, 0.05)
    device = "cuda"
    model = AcDiffusionUNetLab(time_dim)
    # model_files = (
    #     "models/AcDiffusionA.pt",
    #     "models/AcDiffusionB.pt",
    #     "models/AcDiffusionC.pt",
    #     "models/AcDiffusionD.pt",
    #     "models/AcDiffusionE.pt",
    # )
    # load_steps = (200, 400, 600, 800)
    model_files = (
        "models/AcDiffusionLab1A.pt",
        "models/AcDiffusionLab1B.pt"
    )
    load_steps = (500,)
    model.to(device)
    model.eval()
    betas = get_betas(step_num, diff_pars[0], diff_pars[1], diff_pars[2])
    alphas_p = get_alphas_p(step_num, diff_pars[0], diff_pars[1], diff_pars[2], betas)
    for i in range(1, sample_num + 1):
        z = torch.randn([1, 3, H, W], dtype=torch.float32, device=device)
        image = generation(model, z, model_files, load_steps,
                           step_num, time_dim, betas, alphas_p,
                           step_num, 1, "steps", save_steps)
        plt.imsave(save_path + f"/{label}-sample{i}.jpg", tensor2image_np(image))


def img2img():
    H, W = 104, 184
    step_num = 1000
    step_start = 300
    time_dim = 1024
    diff_pars = (0.95, 0.0001, 0.05)
    save_steps = False
    model = AcDiffusionUNet(time_dim)
    model_files = (
        "models/AcDiffusionA.pt",
        "models/AcDiffusionB.pt",
        "models/AcDiffusionC.pt",
        "models/AcDiffusionD.pt",
        "models/AcDiffusionE.pt",
    )
    load_steps = (200, 400, 600, 800)
    # path = "d:/DeepLearning/LL/*"
    path = "/mnt/DF2E413D652A1785/DeepLearning/LL/*"
    # path = "d:/DeepLearning/waifu_test/*"
    # path = "/mnt/DF2E413D652A1785/BaiduNetdiskDownload/CelebA/Img/img_celeba.7z/img_celeba/*"
    # path = "e:/0000/Album/*"
    device = "cuda"
    model.to(device)
    model.eval()
    data = AcDataset(path, W, H)
    data.enable_dynamic_shape(16 * 16, 8)
    print(len(data))
    betas = get_betas(step_num, diff_pars[0], diff_pars[1], diff_pars[2])
    alphas_p = get_alphas_p(step_num, diff_pars[0], diff_pars[1], diff_pars[2], betas)
    image = random.choice(data)
    H, W = image.shape[-2:]
    image = image.view(1, 3, H, W).to(device)
    noise = torch.randn([1, 3, H, W], dtype=torch.float32, device=device)
    image_n = image * np.sqrt(alphas_p[step_start-1]) + noise * np.sqrt(1. - alphas_p[step_start-1])
    
    out = generation(model, image_n, model_files, load_steps,
                     step_num, time_dim, betas, alphas_p,
                     step_start, 1, "steps", save_steps)
    
    plt.subplot(1, 3, 1)
    plt.imshow(tensor2image_np(image))
    plt.subplot(1, 3, 2)
    plt.imshow(tensor2image_np(image_n))
    plt.subplot(1, 3, 3)
    plt.imshow(tensor2image_np(out))
    plt.show()


def test_AcDiffusion():
    H, W = 104, 184
    step_num = 1000
    model_path_A="models/AcDiffusionA.pt"
    model_path_B="models/AcDiffusionB.pt"
    save_steps = False
    time_dim = 1024
    diff_pars = (0.95, 0.0001, 0.05)
    device = "cuda"
    model = AcDiffusionUNet(time_dim)
    model.to(device)
    model.eval()
    betas = get_betas(step_num, diff_pars[0], diff_pars[1], diff_pars[2])
    alphas_p = get_alphas_p(step_num, diff_pars[0], diff_pars[1], diff_pars[2], betas)
    z = torch.randn([1, 3, H, W], dtype=torch.float32, device=device)
    # z = torch.sigmoid(z)
    # noise = torch.randn([1, 3, H, W], dtype=torch.float32, device=device)
    # z = z * np.sqrt(alphas_p[-1]) + noise * np.sqrt(1. - alphas_p[-1])
    model.load_state_dict(torch.load(model_path_B, map_location="cpu"))
    image = generation(model, z,
                       step_num, time_dim, betas, alphas_p,
                       1000, 501, "steps", save_steps)
    model.load_state_dict(torch.load(model_path_A, map_location="cpu"))
    image = generation(model, image,
                       step_num, time_dim, betas, alphas_p,
                       500, 1, "steps", save_steps)
    plt.imshow(tensor2image_np(image))
    plt.show()


def save_samples():
    model_path = "models/16200-AcDiffusion.pt"
    device = "cuda"
    model = AcDiffusionUNet()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    image = torch.randn([1, 3, 128, 128], dtype=torch.float32, device=device)
    image = generation(model, (0.98, 0.0001, 0.02), image, 1000, 1000, 0, "samples", True)
    plt.imshow(image.detach().cpu().view(3, 128, 128).permute(1, 2, 0).clip(0.0, 1.0))
    plt.show()


def show_diff_graph():
    step_num = 100
    x = np.linspace(0, step_num - 1, step_num, dtype=np.float32)
    y = get_betas(step_num, 0.95, 0.0001, 0.05)
    plt.subplot(2, 2, 1)
    plt.plot(x, y)
    
    plt.subplot(2, 2, 2)
    y = get_betas(step_num, 0.9, 0.0001, 0.04)
    plt.plot(x, y)
    
    plt.subplot(2, 2, 3)
    y = get_alphas_p(step_num, 0.95, 0.0001, 0.05)
    plt.plot(x, y)
    
    plt.subplot(2, 2, 4)
    y = get_alphas_p(step_num, 0.9, 0.0001, 0.04)
    plt.plot(x, y)
    plt.show()


def test_diff():
    H, W = 104, 184
    # paths = glob.glob("/mnt/F89B887EA07147CB/DeepLearning/LL/*")
    # image = Image.open(random.choice(paths)).convert("RGB")
    image = Image.open("/mnt/F89B887EA07147CB/DeepLearning/test.png").convert("RGB")
    image = ImageOps.fit(image, (W, H))
    image = np.array(image, dtype=np.float32) / (255.0 / 2.0) - 1.0
    
    H = 1
    W = 5
    step_num = 1000
    alphas_p = get_alphas_p(step_num, 0.9, 0.0001, 0.04)
    # print(alphas_p)
    c = 0
    for h in range(H):
        for w in range(W):
            t = step_num // (H * W) * c
            k1 = np.sqrt(alphas_p[t])
            k2 = np.sqrt(1 - alphas_p[t])
            print(t, k1, k2)
            temp = image * k1 + np.random.randn(*image.shape) * k2
            temp = temp * 0.5 + 0.5
            plt.subplot(H, W, c + 1)
            plt.imshow(temp.clip(0.0, 1.0))
            plt.imsave(f"{h + w}.jpg", temp.clip(0.0, 1.0))
            c += 1
    
    plt.show()
    # temp = image * np.sqrt(alphas_p[-1]) + np.random.randn(*image.shape) * np.sqrt(1 - alphas_p[-1])
    # temp = temp * 0.5 + 0.5
    # plt.imshow(temp.clip(0.0, 1.0))
    # plt.show()


def test_pos_encode():
    dim = 1024
    steps = range(1, 1000+1)
    graph = np.zeros([dim, 1], dtype=np.float32)
    for step in steps:
        t_emb = position_encoding(step, dim).reshape(-1, 1)
        graph = np.concatenate((graph, t_emb), axis=1)
    print(graph.shape)
    plt.imshow(graph)
    plt.show()


def test_AcDataSetForSR():
    H, W = 104, 184
    path = "d:/DeepLearning/LL/*"
    data = AcDatasetForSR(path, W*4, H*4, W, H)
    print(len(data))
    hr, lr = random.choice(data)
    print(hr.shape, lr.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(hr.permute(1, 2, 0).clip(0.0, 1.0))
    plt.subplot(1, 2, 2)
    plt.imshow(lr.permute(1, 2, 0).clip(0.0, 1.0))
    plt.show()


def test_PIL():
    path = "/mnt/F89B887EA07147CB/DeepLearning/LL/*"
    paths = glob.glob(path)
    image = Image.open(random.choice(paths)).convert("RGB")
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.subplot(2, 2, 2)
    W, H = image.size
    d = (random.random() * 2.0 - 1.0) * 10.0
    r = abs(d) * math.pi / 180.0
    k = (1.0 / math.tan(r)) + (1.0 / math.sin(r))
    t = k * k - 1.0
    l = max((k * W - H) / t, (k * H - W) / t)
    f = H / W
    offset_w = int(l / (f + math.tan(r))) + 1
    offset_h = int(f * offset_w) + 1
    print(H, W, d, math.pi, r, k, l, offset_h, offset_w)
    image = image.rotate(d, resample=Image.Resampling.BILINEAR)
    plt.imshow(image)
    plt.subplot(2, 2, 3)
    image = image.crop((offset_w, offset_h, W - offset_w, H - offset_h))
    W, H = image.size
    print(W, H)
    plt.imshow(image)
    plt.subplot(2, 2, 4)
    pixel_drop_w = random.randint(0, 200)
    pixel_drop_h = H - int((W - pixel_drop_w) * f)
    if pixel_drop_h < 0:
        pixel_drop_h = 0
    W -= pixel_drop_w
    H -= pixel_drop_h
    offset_w = random.randint(0, pixel_drop_w)
    offset_h = random.randint(0, pixel_drop_h)
    image = image.crop((offset_w, offset_h, offset_w + W, offset_h + H))
    print(W, H, pixel_drop_w, pixel_drop_h, offset_w, offset_h)
    plt.imshow(image)
    plt.show()


def test_group_conv():
    times = 200
    c = 64
    f_size = 128
    groups = c // 16 if c >= 16 else 1
    batch_size = 4
    device = "cuda"
    x = torch.randn([batch_size, 3, f_size, f_size], dtype=torch.float32, device=device)
    conv = nn.Conv2d(3, c, 3, 1, 1).to(device)
    group_convs = []
    kernel_sizes = range(5, 30, 2)
    for kernel_size in kernel_sizes:
        group_convs.append(
            nn.Conv2d(3, c, kernel_size, 1, (kernel_size-1)//2, groups=1).to(device)
        )
    print("Warming up...")
    for _ in range(times // 2):
        # x = conv(x).detach()
        y = conv(x).detach()
        for conv in group_convs:
            # x = conv(x).detach()
            y = conv(x).detach()
    print("Testing...")
    t = time.time()
    for _ in range(times):
        # x = conv(x).detach()
        y = conv(x).detach()
    time_conv = (time.time() - t) / times
    print(f"time_conv: {time_conv}")
    time_gc_scale = []
    for i, conv in enumerate(group_convs):
        t = time.time()
        for _ in range(times):
        #    x = conv(x).detach()
           y = conv(x).detach()
        scale = (time.time() - t) / times / time_conv
        print(f"kernel_size: {kernel_sizes[i]}, scale: {scale}")
        time_gc_scale.append(scale)
    plot_conv = np.ones([len(time_gc_scale)], dtype=np.float32)
    plt.plot(kernel_sizes, plot_conv)
    plt.plot(kernel_sizes, time_gc_scale)
    plt.show()


def show_losses():
    losses = np.load("models/AcDiffusionLab1A.pt.npy")
    print(len(losses))
    steps = np.linspace(1, len(losses), len(losses), dtype=np.int32)
    plt.plot(steps, losses)
    plt.show()


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(1,1, bias=False),
            nn.Tanh(),
            nn.Linear(1,1, bias=False),
            nn.Tanh(),
            nn.Linear(1,1, bias=False),
            nn.Tanh(),
            nn.Linear(1,1, bias=False),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.model(x)


def test3():
    x = torch.tensor([[1, 2, 3]], dtype=torch.float32)
    x = x[:, :, None, None].repeat(2, 1, 4, 3)
    print(x)
    print(x.size())
    
    model = TestModel()
    for param in model.parameters():
        param.data = torch.tensor([[1]], dtype=torch.float32)
    print("param=%s" % (param.data.item()))
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    predict_y = model(torch.tensor([1], dtype=torch.float32))
    loss = loss_func(predict_y, torch.tensor([2], dtype=torch.float32))
    model.zero_grad()
    loss.backward()
    for param in model.parameters():
        print("param=%s, grad=%s" % (param.data.item(), param.grad.item()))
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=2)
    for param in model.parameters():
        print("param=%s, grad=%s" % (param.data.item(), param.grad.item()))
    optimizer.step()
    for param in model.parameters():
        print("param=%s, grad=%s" % (param.data.item(), param.grad.item()))
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5,
                                                    total_steps=1000,
                                                    div_factor=10.,
                                                    final_div_factor=1.)
    lr_list = []
    for e in range(10):
        for i in range(100):
            optimizer.zero_grad()
            optimizer.step()
            lr_list.append(optimizer.param_groups[0]['lr'])
            if e == 0 and i < 10:
                print(optimizer.param_groups[0]['lr'])
            elif e == 9 and i > 790:
                print(optimizer.param_groups[0]['lr'])
            scheduler.step()
    plt.plot(range(1000), lr_list)
    plt.show()


def LR():
    paths = glob.glob("samples/dataset/*")
    c = 1
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = ImageOps.fit(image, (184, 104))
        image.save(f"samples/dataset-LR/{c}.jpg")
        c += 1


def show_losses():
    losses = np.load(f"models/AcDiffusionV0A.pt.npy")
    losses = losses[:10000]
    print(len(losses))
    steps = np.linspace(1, len(losses), len(losses), dtype=np.int32)
    plt.plot(steps, losses)
    plt.show()


def activation():
    x = torch.linspace(-5., 5., 1000, dtype=torch.float32, device="cpu")
    zeros = torch.zeros_like(x)
    y1 = torch.atan(x)
    datan = 1.0 / (x**2 + 1.0)
    y2 = torch.tanh(x)
    y3 = x / (x ** 2 * 0.25 + 1.)
    plt.plot(x, x)
    plt.plot(x, zeros)
    # plt.plot(x, y1)
    # plt.plot(x, y2)
    plt.plot(x, y3)
    # plt.plot(x, datan)
    plt.show()


if __name__ == '__main__':
    # test2()
    # test_SR()
    # test_SR_sample()
    # test_discriminator()
    # summary_SR()
    # summary_D()
    # summary_U()
    # test2()
    # test_AcDataSetForDiff()
    # test_AcDiffusion()
    # save_samples()
    # show_diff_graph()
    # test_diff()
    # sample()
    # img2img()
    # test_pos_encode()
    # test_AcDataSetForSR()
    # test_PIL()
    # test_group_conv()
    # show_losses()
    # test3()
    # LR()
    # show_losses()
    activation()
