from utils import *
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm
import time
from torchinfo import summary
from torchcodec.decoders import VideoDecoder

from yolo_v12 import YoloV12mBackbone


class _Dataset(Dataset):
    def __init__(self, size, dtype=torch.float32, test=False, length=1000) -> None:
        super().__init__()
        self.length = length
        path_list = [
            "/home/tobyh/work-space/datasets/anime"
        ]
        file_list = []
        self.decorders = []
        for p in path_list:
            file_list += glob.glob(f"{p}/*.mp4")
            file_list += glob.glob(f"{p}/*.mkv")
        for f in file_list:
            self.decorders.append(VideoDecoder(f))
        print(f"_Dataset: len(self.decorders) == {len(self.decorders)}")
        exit(0)

        # self.paths = glob.glob(path)
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
        self.random = True
        self.s = 0
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, _):
        # image = Image.open(self.paths[index]).convert("RGB")
        i = random.randint(0, len(self.vr_list) - 1)
        image = self.vr_list[i][random.randint(0, len(self.vr_list[i]) - 1)]
        # image = torchvision.io.read_image(self.paths[index], torchvision.io.ImageReadMode.RGB)
        image = self.transforms(image).mul_(2.0).add_(-1.0)
        noise = torch.randn_like(image)
        d = noise - image
        if self.random:
            s = random.random()
        else:
            s = self.s
        return image.add_(d.mul_(s)), torch.tensor([math.sqrt(s)], dtype=image.dtype)
    
    def test(self, plot_h, plot_w):
        self.random = False
        for h in range(plot_h):
            for w in range(plot_w):
                i = plot_w * h + w
                self.s = 1 / (plot_h*plot_w-1) * i
                image, s = random.choice(self)
                print(s)
                plt.subplot(plot_h, plot_w, i + 1)
                plt.imshow(tensor2image_np(image))
        plt.show()


class _GradModel(nn.Module):
    def __init__(self, size=(9*64, 16*64), c=0, act="ELU"):
        super().__init__()
        print("GradModel initializing...")
        C_IN = 32
        S = 128
        C = (S, S*2, S*4)
        C_Z = 64
        block1 = {
            "c": C[0],
            "activation": act,
            "layers": "rrrr"
        }
        block2 = {
            "c": C[1],
            "activation": act,
            "layers": "rrrr",
            # "layers": "rrsrrsrr",
            # "num_heads": (4,)
        }
        H, W = size

        self.nn = nn.Sequential(
            nn.Conv2d(3, C_IN, 4, 2, 1),
            activition(act, True),
            nn.Conv2d(C_IN, C[0], 4, 2, 1),
            Block(block1, ("down", C[1])),
            Block(block2, ("down", C[2])),
            Conv(C[2], C_Z, 1, 1, 0, act),
            activition(act, True),
            nn.Flatten()
        )
        
        self.D = nn.Sequential(
            nn.Linear(32*(H//64)*(W//64), 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.nn(x)

    def summary(self):
        H, W = 9*64, 16*64
        device, dtype = "cuda", torch.float32
        self.eval().to(device, dtype)
        z = torch.randn([1, 3, H, W], dtype=dtype, device=device)
        looper = tqdm(range(100))
        t = time.time()
        for _ in looper:
            _ = self(z.detach())
        t = time.time() - t
        print(f"{100. / t} /s")
        summary(self, input_size=(1, 3, H, W))


def train():
    name = "yolov12m"
    epoch = 20
    batch_size = 8
    gradient_accumulation_steps = 2
    learning_rate = 2e-5
    img_lr = 100.0
    img_step = 100
    save_epoch = 5
    device, dtype = "cuda", torch.float32
    H, W = 9*64, 16*64

    os.makedirs(f"models", exist_ok=True)
    os.makedirs(f"samples/{name}", exist_ok=True)
    if epoch < save_epoch:
        save_epoch = epoch
    epoch_exist = len(glob.glob(f"samples/{name}/epoch*")) * save_epoch
    dataset = _Dataset((H, W), dtype)
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, drop_last=True)
    model = YoloV12mBackbone((H, W)).train().to(device=device, dtype=dtype)
    # model = _GradModel((H, W))
    load_model_file(model, f"{name}.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        total_steps=epoch*len(loader),
        div_factor=10.0,
        final_div_factor=1.0
    )
    loss_list = []
    tt = time.time()
    for e in range(1, epoch + 1):
        looper = tqdm(loader, total=len(loader))
        for data in looper:
            images = data[0].to(device).requires_grad_(False)
            scores = data[1].to(device).requires_grad_(False)
            # print(scores)
            # for i in range(4):
            #     print(scores[i])
            #     plt.subplot(2, 2, i+1)
            #     plt.imshow(tensor2image_np(images[i]))
            # plt.show()
            # exit(0)
            optimizer.zero_grad()
            out, _ = model(images)
            loss = F.l1_loss(out, scores)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loss_list.append(loss.item())
            looper.set_description(f"epoch {e}")
            looper.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        if e % save_epoch == 0:
            torch.save(model.state_dict(), f"models/{name}.pt")
            images = torch.randn_like(images, requires_grad=True)
            looper = tqdm(range(img_step), total=img_step)
            looper.set_description(f"sampling")
            for _ in looper:
                out, _ = model(images)
                loss = F.l1_loss(out, torch.zeros_like(out, requires_grad=False))
                loss.backward()
                images = images.detach_().add_(images.grad.mul_(-img_lr))
                images = images.detach_().requires_grad_(True)
                images.grad.mul_(0)
                looper.set_postfix(loss=loss.item())
            utils.save_image(
                images,
                f"samples/{name}/epoch-{epoch_exist + e}.png",
                nrow=images.shape[0],
                normalize=True
            )
    
    print(f"Train time: {time.time() - tt}")
    plt.plot(range(1, len(loss_list)+1), loss_list)
    fig_exist = len(glob.glob(f"samples/{name}/Figure*"))
    plt.savefig(f"samples/{name}/Figure{fig_exist+1}.png")


def sample():
    name = "yolov12m"
    batch_size = 4
    img_lr = 1000
    max_step = 1000
    device, dtype = "cuda", torch.float32
    H, W = 9*64, 16*64

    model = YoloV12mBackbone((H, W)).train().to(device=device, dtype=dtype)
    load_model_file(model, f"{name}.pt")
    images = torch.randn([batch_size, 3, H, W], device=device, dtype=dtype, requires_grad=True)
    looper = tqdm(range(max_step), total=max_step)
    looper.set_description(f"sampling")
    for _ in looper:
        out, _ = model(images)
        loss = F.l1_loss(out, torch.zeros_like(out, requires_grad=False))
        if loss.item() < 0.1:
            break
        loss.backward()
        grad_min, grad_max = images.grad.min(), images.grad.max()
        images = images.detach_().add_(images.grad.mul_(-img_lr))
        # images.add_(torch.randn_like(images).mul_(loss**2*0.1))
        images = images.detach_().requires_grad_(True)
        images.grad.mul_(0)
        looper.set_postfix(
            loss=loss.item(), grad_min=grad_min.item(),
            grad_max=grad_max.item(), score_mean=out.mean().item()
        )
    img_exist = len(glob.glob(f"samples/{name}/sample-*"))
    utils.save_image(
        images,
        f"samples/{name}/sample-{img_exist + 1}.png",
        nrow=images.shape[0],
        normalize=True
    )


def test1():
    x = torch.linspace(0, 1.0, 1000, dtype=torch.float32)
    y = x.pow(0.5)
    plt.plot(x, y)
    plt.plot(x, x)
    plt.show()


if __name__ == '__main__':
    # train()
    # sample()
    # test1()

    # m = _GradModel()
    # m = YoloV12mBackbone()
    # m.summary()

    dataset = _Dataset((9*64, 16*64))
    print(len(dataset))
    dataset.test(3, 4)
