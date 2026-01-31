import decord
from tqdm import tqdm
import random
import matplotlib.pylab as plt


dir = "/run/media/tobyh/00421E99421E940E/000000/Dandadan"
file = "[Nekomoe kissaten&LoliHouse] Dandadan - 12 [WebRip 1080p HEVC-10bit AAC ASSx2].mkv"
save_dir = "/home/tobyh/work-space/datasets/dandadan"


def video_to_frames(num=500):
    decord.bridge.set_bridge('torch')
    decoder = decord.VideoReader(f"{dir}/{file}")
    picks = []
    for _ in range(num):
        pick = random.randint(0, len(decoder)-1)
        while pick in picks:
            pick = random.randint(0, len(decoder)-1)
        picks.append(pick)
    picks.sort()
    looper = tqdm(picks, total=num)
    for pick in looper:
        plt.imsave(f"{save_dir}/{file}[{pick:05d}].png", decoder[pick])


if __name__ == '__main__':
    video_to_frames()
