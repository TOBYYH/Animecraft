from utils import *
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm
import time
from torchinfo import summary
from torchvision.transforms import InterpolationMode


# PATH = "/home/tobyh/work-space/datasets/anime/*"
PATH = "/home/tobyh/work-space/datasets/dandadan/*"
NAME = "diffusion"
SIZE = (10*32, 16*32)


class _Dataset(Dataset):
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
                v2.RandomResizedCrop(
                    size, (0.5, 1.0), (ratio*0.9, ratio*1.1), InterpolationMode.BICUBIC
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.ToDtype(dtype, scale=True)
            ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        # image = Image.open(self.paths[index]).convert("RGB")
        image = torchvision.io.read_image(self.paths[index], torchvision.io.ImageReadMode.RGB)
        image = self.transforms(image).mul_(2.0).add_(-1.0)
        noise = torch.randn_like(image)
        d = image - noise
        k = random.random()
        return image.add_(-d * k), d, torch.tensor([k], dtype=image.dtype)
    
    def test(self, plot_h, plot_w):
        for h in range(plot_h):
            for w in range(plot_w // 2):
                image, d, s = random.choice(self)
                print(s)
                plt.subplot(plot_h, plot_w, plot_w*h+w*2+1)
                plt.imshow(tensor2image_np(image))
                plt.subplot(plot_h, plot_w, plot_w*h+w*2+2)
                plt.imshow(tensor2image_np(d))
        plt.show()


def auto_pad(k):
    return (k - 1) // 2


class _ResBlock(nn.Module):
    def __init__(self, c, c2, k1, k2, g=1, act="tanh") -> None:
        super().__init__()
        self.nn = nn.Sequential(
            group_norm(c),
            nn.Conv2d(c, c2, k1, 1, auto_pad(k1), groups=g),
            activition(act, True),
            nn.Conv2d(c2, c, k2, 1, auto_pad(k2), groups=g)
        )
    
    def forward(self, x):
        return self.nn(x) + x


class _Down(nn.Module):
    def __init__(self, c_in, c_out, k, g:int=1, act="tanh"):
        super().__init__()
        self.down = nn.Sequential(
            group_norm(c_in),
            nn.Conv2d(c_in, c_out, 4, 2, 1)
        )
        self.res = nn.Sequential(
            _ResBlock(c_out, c_out, 5, 1, act=act),
            _ResBlock(c_out, c_out, 3, 3, act=act)
        )
    
    def forward(self, x):
        return self.res(self.down(x))


class _Up(nn.Module):
    def __init__(self, c_in, c_out, k, g:int=1, act="tanh"):
        super().__init__()
        self.down = nn.Sequential(
            group_norm(c_in),
            nn.ConvTranspose2d(c_in, c_out, 4, 2, 1)
        )
        self.res = nn.Sequential(
            _ResBlock(c_out, c_out, 5, 1, act=act),
            _ResBlock(c_out, c_out, 3, 3, act=act)
        )
    
    def forward(self, x):
        return self.res(self.down(x))


class _Model(nn.Module):
    def __init__(self, size, c=0, act="ELU"):
        super().__init__()
        self.size = size
        H, W = size
        C = [16, 64, 128, 256, 512]
        in_block = nn.Conv2d(3, C[0], 3, 1, 1)
        self.nn_list = nn.ModuleList([
            in_block,
            _Down(C[0], C[1], 3, act=act),
            _Down(C[1], C[2], 3, act=act),
            _Down(C[2], C[3], 3, act=act),
            _Down(C[3], C[4], 3, act=act),
            Conv(C[4], C[4]//2, 4, 2, 1, act=act),
            Conv(C[4]//2, C[4]//4, 4, 2, 1, act=act),
            nn.Flatten(),
            nn.Linear(C[4]//4*(H//64)*(W//64), 1),
            nn.Sigmoid(),
            _Up(C[4], C[3], 3, act=act),
            _Up(C[3]*2, C[2], 3, act=act),
            _Up(C[2]*2, C[1], 3, act=act),
            _Up(C[1]*2, C[0], 3, act=act),
            nn.Conv2d(C[0]*2, 3, 3, 1, 1)
        ])
    
    def forward(self, x):
        v_list = []
        z = x
        for i in range(4):
            out = self.nn_list[i](z)
            v_list.append(out)
            z = out
        z = self.nn_list[4](z)
        v1 = z
        for i in range(5, 10):
            z = self.nn_list[i](z)
        k = z
        z = self.nn_list[10](v1)
        for i in range(11, len(self.nn_list)):
            z = torch.cat([z, v_list[10 - i]], dim=1)
            z = self.nn_list[i](z)
        return z, k
    
    def summary(self):
        H, W = self.size
        n = 100
        device, dtype = "cuda", torch.float32
        self.eval().to(device, dtype)
        z = torch.randn([1, 3, H, W], dtype=dtype, device=device)
        looper = tqdm(range(n))
        t = time.time()
        for _ in looper:
            v, _ = self(z.detach())
        t = time.time() - t
        print(f"{n / t} /s")
        print(v.shape)
        summary(self, input_size=(1, 3, H, W))


def sample(model, step_num, lr, shape, dtype, device):
    N, C, H, W = shape
    assert C == 3
    k_mean = 1.0
    with torch.no_grad():
        images = torch.randn(shape, dtype=dtype, device=device)
        looper = tqdm(range(step_num), total=step_num)
        looper.set_description(f"sampling")
        for i in looper:
            d, k = model(images)
            m = k.mean().item()
            looper.set_postfix(k_mean=m)
            # if i > 10 and m > k_mean:
            #     break
            k_mean = m
            if i == step_num - 1:
                lr = 1.0
            images.add_(
                d.mul_(k.mul(-lr).view(N, 1, 1, 1).repeat(1, 3, H, W))
            )
            # images.add_(d.mul_(k.view(N, 1, 1, 1).repeat(1, 3, H, W)))
            # images.add_(torch.randn_like(images).mul_(1.0/step_num*(step_num-1-i)))
    return images


def train():
    epoch = 100
    save_epoch = 10
    batch_size = 14
    gradient_accumulation_steps = 2
    learning_rate = 1e-5
    sample_step = 20
    sample_lr = 0.1
    device, dtype = "cuda", torch.float32

    os.makedirs(f"models", exist_ok=True)
    os.makedirs(f"samples/{NAME}", exist_ok=True)
    if epoch < save_epoch:
        save_epoch = epoch
    epoch_exist = len(glob.glob(f"samples/{NAME}/epoch*")) * save_epoch
    dataset = _Dataset(PATH, SIZE, dtype)
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2, drop_last=True)
    model = _Model(SIZE).train().to(device=device, dtype=dtype)
    load_model_file(model, f"{NAME}.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        total_steps=epoch*len(loader),
        div_factor=10.0,
        final_div_factor=0.5
    )
    loss_list = []
    tt = time.time()
    for e in range(1, epoch + 1):
        looper = tqdm(loader, total=len(loader))
        for data in looper:
            images = data[0].detach_().to(device).requires_grad_(False)
            d = data[1].detach_().to(device).requires_grad_(False)
            k = data[2].detach_().to(device).requires_grad_(False)
            # print(k)
            # for i in range(4):
            #     print(k[i])
            #     plt.subplot(2, 2, i+1)
            #     plt.imshow(tensor2image_np(images[i]))
            # plt.show()
            # exit(0)
            optimizer.zero_grad()
            out_d, out_k = model(images)
            loss_img = F.mse_loss(out_d, d)
            loss_k = F.mse_loss(out_k, k)
            loss = loss_img + loss_k
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_list.append(loss.item())
            looper.set_description(f"epoch {e}")
            looper.set_postfix(
                loss=loss.item(), loss_img=loss_img.item(),
                loss_k=loss_k.item(), lr=optimizer.param_groups[0]['lr']
            )
        
        if e % save_epoch == 0:
            torch.save(model.state_dict(), f"models/{NAME}.pt")
            images = sample(
                model, sample_step, sample_lr, images.shape, images.dtype, images.device
            )
            utils.save_image(
                images,
                f"samples/{NAME}/epoch-{epoch_exist + e}.png",
                nrow=images.shape[0]//4,
                normalize=True
            )
    
    print(f"Train time: {time.time() - tt}")
    plt.plot(range(1, len(loss_list)+1), loss_list)
    fig_exist = len(glob.glob(f"samples/{NAME}/Figure*"))
    plt.savefig(f"samples/{NAME}/Figure{fig_exist+1}.png")


def generate():
    device, dtype = "cuda", torch.float32
    H, W = SIZE
    model = _Model(SIZE).eval().to(device=device, dtype=dtype)
    load_model_file(model, f"{NAME}.pt")
    images = sample(model, 100, 0.1, [4, 3, H, W], dtype, device)
    img_exist = len(glob.glob(f"samples/{NAME}/sample-*"))
    utils.save_image(
        images,
        f"samples/{NAME}/sample-{img_exist + 1}.png",
        nrow=images.shape[0],
        normalize=True
    )


if __name__ == '__main__':
    # train()
    generate()

    # dataset = _Dataset(PATH, SIZE)
    # print(len(dataset))
    # dataset.test(3, 4)

    # m = _Model(SIZE)
    # m.summary()
