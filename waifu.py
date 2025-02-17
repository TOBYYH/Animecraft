import decord
from tqdm import tqdm
import glob
import matplotlib.pylab as plt
import torchvision.transforms as T


def video_to_frames():
    decord.bridge.set_bridge('torch')
    
    # dir = "D:/Downloads/[Moozzi2] Mushoku Tensei II Isekai Ittara Honki Dasu [ x265-10Bit Ver. ] - TV + SP"
    # file = "[Moozzi2] Mushoku Tensei II Isekai Ittara Honki Dasu - 24 END (BD 1920x1080 x265-10Bit Flac).mkv"
    dir = "F:/000000/[DBD-Raws][无职转生 ~到了异世界就拿出真本事~][01-23TV全集+OVA][1080P][BDRip][HEVC-10bit][简繁外挂][FLAC][MKV]"
    file = "[DBD-Raws][无职转生 ~到了异世界就拿出真本事~][02][1080P][BDRip][HEVC-10bit][FLAC].mkv"
    vr = decord.VideoReader(f"{dir}/{file}")

    start, end, save_num = 100, len(vr), 400
    # start, end, save_num = 23900, 24100, 10

    print('video frames:', len(vr))
    id_list = range(start, end, (end - start) // save_num)
    looper = tqdm(id_list, total=len(id_list))
    for i in looper:
        plt.imsave(f"frames/{file}-{i}.png", vr[i].numpy())


if __name__ == '__main__':
    video_to_frames()
