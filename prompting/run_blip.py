import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
from decord import VideoReader, cpu, gpu
import argparse
from glob import glob
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default='/gallery_louvre/dayoon.ko/dataset/videohumor/*', type=str)
    parser.add_argument('--num_video', default=10, type=int)
    parser.add_argument('--store_cap', default=False, action='store_true')
    parser.add_argument('--dest_dir', default='/gallery_louvre/dayoon.ko/dataset/blip_intern/blip', type=str)
    parser.add_argument('--num', default=1000, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--inter', default=0, type=int)
    return parser.parse_args()
    

def load_model(device):
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2", model_type="coco", is_eval=True, device=device
    )
    return model, vis_processors


def generate_captions(images, model, vis_processors, device, iter=20):
    # image is PIL
    images = torch.stack([vis_processors['eval'](img) for img in images]).to(device)
    caps = []
    for i in range(iter):
        out = model.generate({'image': images, "prompt": "Question: Who is doing what? Answer:"}, use_nucleus_sampling=True)
        caps.append(out)
    caps = [ [caps[i][b] for i in range(len(caps))] for b in range(len(caps[0]))]
    return caps


def caption_video(video_pth, model, vis_processors, device, sample_fps=5.0, num_captions=2):
    vr = VideoReader(video_pth, ctx=cpu(0))
    org_fps = vr.get_avg_fps()
    len_frames = vr[:].shape[0]
    t_stride = int(round(float(org_fps)/float(sample_fps)))
    images = [Image.fromarray(vr[f].numpy()) for f in range(0, len_frames, t_stride)]
    caps = generate_captions(images, model, vis_processors, device, num_captions)
    return caps
        