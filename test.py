from utils import *
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import utils
import clip


def test1():
    x = torch.linspace(-5., 5., 1000, dtype=torch.float32)
    print(x)
    # y = x / (x ** 2 * 0.25 + 1.)
    y = 1.0 / (torch.exp(x) + 1.0)
    plt.plot(x, y)
    plt.plot(x, x)
    plt.show()


class TestDataset(Dataset):
    def __init__(self, path, size) -> None:
        super().__init__()
        self.paths = glob.glob(path)
        self.transforms = v2.Compose([
            v2.Resize(size),
            v2.ToDtype(torch.float32, scale=True)
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        # image = Image.open(self.paths[index]).convert("RGB")
        image = torchvision.io.read_image(self.paths[index], torchvision.io.ImageReadMode.RGB)
        return self.transforms(image)


class AutoEncoder(nn.Module):
    def __init__(self, k, p, g):
        super().__init__()
        C = (16, 64, 128, 256)
        layer_list = [
            nn.Conv2d(3, C[0], 4, 2, 1),
            nn.Tanh()
        ]
        for i in range(len(C)):
            if i > 0:
                layer_list += [
                    nn.Conv2d(C[i-1], C[i], 4, 2, 1),
                    nn.Tanh()
                ]
            layer_list += [
                nn.Conv2d(C[i], C[i], k, 1, p, groups=g),
                nn.Tanh(),
                nn.Conv2d(C[i], C[i], k, 1, p, groups=g),
                nn.Tanh()
            ]
        for i in range(len(C) - 1, 0, -1):
            layer_list += [
                nn.ConvTranspose2d(C[i], C[i-1], 4, 2, 1),
                nn.Tanh(),
                nn.Conv2d(C[i-1], C[i-1], k, 1, p, groups=g),
                nn.Tanh(),
                nn.Conv2d(C[i-1], C[i-1], k, 1, p, groups=g),
                nn.Tanh()
            ]
        layer_list.append(nn.ConvTranspose2d(C[0], 3, 4, 2, 1))
        
        self.layers = nn.Sequential(*layer_list)
    
    def forward(self, x):
        return self.layers(x)


def test2():
    epoch = 5
    device = "cuda"
    H, W = 9*64, 16*64
    dataset = TestDataset("../dataset/LL/*", (H, W))
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8, drop_last=True)
    model = AutoEncoder(3, 1, 1).train().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_list = []
    tt = time.time()
    for e in range(1, epoch + 1):
        looper = tqdm(loader, total=len(loader))
        for data in looper:
            images = data.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(images.detach()), images)
            loss.backward()
            optimizer.step()
            looper.set_description(f"epoch {e}")
            looper.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            loss_list.append(loss.item())
    
    print(f"Train time: {time.time() - tt}")
    model.eval()
    with torch.no_grad():
        out = model(images)
        utils.save_image(
            torch.cat([images, out], 0),
            f"samples/AutoEncoder.png",
            nrow=images.shape[0],
            normalize=True
        )
    plt.plot(range(1, len(loss_list)+1), loss_list)
    plt.show()


def test3():
    times = 100
    device = "cuda"
    H, W = 9*64, 16*64
    model = AutoEncoder((1, 11), (0, 5), 1).eval().to(device)
    z = torch.randn([8, 3, H, W], dtype=torch.float32, device=device)
    looper = tqdm(range(times))
    t = time.time()
    for _ in looper:
        z = model(z.detach())
    print(f"{times / (time.time() - t)} it/s")


def test_conv_channel_num():
    time_scale = 10000
    kernel_size = 3
    f_size = 128
    batch_size = 4
    device = "cuda"
    c_list = range(300, 400)
    time_list = [time_scale // c for c in c_list]
    convs = []
    for c in c_list:
        convs.append(
            nn.Conv2d(c, c, kernel_size, 1, (kernel_size-1)//2).to(device)
        )
    result_list = []
    print("Warming up...")
    for i, conv in enumerate(convs):
        x = torch.randn([batch_size, c_list[i], f_size, f_size], dtype=torch.float32, device=device)
        for _ in range(time_list[i]//2):
            y = conv(x).detach()
    print("Testing...")
    for i, conv in enumerate(convs):
        x = torch.randn([batch_size, c_list[i], f_size, f_size], dtype=torch.float32, device=device)
        t = time.time()
        for _ in range(time_list[i]):
            y = conv(x).detach()
        result_list.append((time.time() - t) / time_list[i])
    plt.plot(c_list, result_list)
    plt.show()


class SimpleConv(nn.Module):
    def __init__(self, c, k, p, g):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(c, c, k, 1, p, groups=g),
            nn.Conv2d(c, c, k, 1, p, groups=g),
            nn.Conv2d(c, c, k, 1, p, groups=g),
            nn.Conv2d(c, c, k, 1, p, groups=g),
            nn.Conv2d(c, c, k, 1, p, groups=g),
            nn.Conv2d(c, c, k, 1, p, groups=g),
            nn.Conv2d(c, c, k, 1, p, groups=g),
            nn.Conv2d(c, c, k, 1, p, groups=g),
            nn.Conv2d(c, c, k, 1, p, groups=g),
            nn.Conv2d(c, c, k, 1, p, groups=g)
        )
    
    def forward(self, x):
        return self.nn(x)


def test4():
    times = 20
    device = "cuda"
    N, C, H, W = 8, 512, 9*4, 16*4
    for k, p in [(1, 0), (3, 1), (5, 2), (7, 3)]:
        for g in [1, 2, 4, 8, 16, 32, 64]:
            model = SimpleConv(C, k, p, g).eval().to(device)
            z = torch.randn([N, C, H, W], dtype=torch.float32, device=device)
            for _ in range(times//2):
                z = model(z.detach())
            t = time.time()
            for _ in range(times):
                z = model(z.detach())
            speed = times / (time.time() - t)
            if isinstance(k, tuple):
                size = k[0] * k[1]
            else:
                size = k * k
            e = speed * size * (C/g)**2 * g / 10**7
            print(f"({k}, {g}) {speed:.2f}it/s {e}")


def test5():
    path = "frames/*.png"
    H, W = 9*64, 16*64
    dataset = AcDataset(path, (H, W), test=True)
    image = random.choice(dataset)
    print(image.shape, image.dtype)
    img_fft = torch.fft.fft2(image)
    print(img_fft.shape, img_fft.dtype)
    show_fft = torch.fft.fftshift(torch.log(torch.abs(img_fft)))
    print(show_fft.shape, show_fft.dtype)
    print(show_fft.max(), show_fft.min(), show_fft.mean())
    show_fft = (show_fft - show_fft.min()) / (show_fft.max() - show_fft.min())
    plt.subplot(2, 2, 1)
    plt.imshow(tensor2image_np(image))
    plt.subplot(2, 2, 2)
    plt.imshow(tensor2image_np(show_fft))
    img_fft = torch.view_as_real(img_fft)
    print(img_fft.shape, img_fft.dtype)
    plt.show()


def lr_scheduler():
    model = AutoEncoder(3, 1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5,
                                                    total_steps=1000,
                                                    div_factor=10.,
                                                    final_div_factor=0.5)
    lr_list = []
    for _ in range(1000):
        lr_list.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    print(lr_list[:10], lr_list[-10:])
    plt.plot(range(1000), lr_list)
    plt.show()


def test_clip():
    device = "cuda"
    print(clip.available_models())
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    image = preprocess(Image.open("img1.jpg")).unsqueeze(0).to(device)
    print(image.shape) # [1, 3, 336, 336]
    texts = [
        "cute", "beautiful", "big", "girl", "happy", "christmas",
        "spring", "summer", "autumn", "winter",
        "three anime girls", "two boys", "merry christmas"
    ]
    text = clip.tokenize(texts).to(device)
    with torch.no_grad():
        tt = time.time()
        image_features = model.encode_image(image.to(device))
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        print(f"{time.time() - tt}s")
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print(image_features.shape)
        print(text_features.shape)
        for i in range(len(texts)):
            print(texts[i], logits_per_text[i, 0].item())
    # print("Label probs:", probs)


if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    # test_conv_channel_num()
    # test4()
    # test5()
    # lr_scheduler()
    test_clip()
