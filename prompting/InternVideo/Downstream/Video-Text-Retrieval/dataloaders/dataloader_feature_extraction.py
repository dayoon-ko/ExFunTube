from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import io
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
from dataloaders.rawvideo_util import RawVideoExtractor
try:
    from petrel_client.client import Client

    client = Client()

    # Disable boto logger
    import logging
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('nose').setLevel(logging.WARNING)
except:
    client = None
    
from decord import VideoReader, cpu
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from glob import glob 
import cv2    
import librosa
from pathlib import Path
from PIL import Image

    
class VideoHumorDataset(Dataset):
    """MSRVTT dataset loader."""
    def __init__(   
                    self,
                    max_words=30,
                    feature_framerate=1.5,
                    max_frames=16,
                    image_resolution=(224, 224),
                    num=0,
                    start=0,
                ):
        self.viddirs = sorted([i for i in list(Path('/gallery_louvre/dayoon.ko/dataset/videohumor').glob('*'))[num*5100 + start: (num+1)*5100]
                               #if not os.path.exists(f'/gallery_louvre/dayoon.ko/dataset/videohumor_maf/video_feats_iv/{i.name}.pt')
                               ], 
                              key=lambda x: str(x).lower())
        self.videos = [glob(f'{p}/*.mp4')[0] for p in self.viddirs]
        
        self.feature_framerate = feature_framerate
        self.max_frames = max_frames
        self.image_resolution = image_resolution

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)

        self.transform = Compose([
                    Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(image_resolution),
                    lambda image: image.convert("RGB"),
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    def __len__(self):
        return len(self.videos)

    def pad_seq(self, frames):
        last = frames[-1]
        pad = torch.zeros((16, *last.shape[1:])) # 16, 3, 224, 224
        for i in range(len(last)):
            pad[i] = last[i]
        frames[-1] = pad
        print(frames[-1].shape)
        return frames
    
    def _get_rawvideo_dec(self, vreader, t_stride, window_length=16):
        # speed up video decode via decord.
        # video_mask = np.zeros(self.max_frames, dtype=np.long)
        frames = []
        for i in range(0, len(vreader), t_stride):
            if i + window_length >= len(vreader):
                continue
            frames.append(torch.stack([self.transform(Image.fromarray(img)) for img in vreader[i:i+window_length].asnumpy()]))
        print(len(frames))
        #frames = self.pad_seq(frames)
        video = torch.stack(frames)
        video = video.unsqueeze(2) # to make size pair, block_size, time_segment(1), c(3), h, w
            
        video_mask = torch.ones(video.shape[:3])
        return video.unsqueeze(1), video_mask
    
    
    def __getitem__(self, idx):
        vid = self.viddirs[idx].name
        video_path = self.videos[idx]
        video_cap = cv2.VideoCapture(video_path)
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        hop_size = int(fps * self.feature_framerate)
        
        vreader = VideoReader(video_path, ctx=cpu(0))
        video, video_mask = self._get_rawvideo_dec(vreader, hop_size)
        return vid, video, video_mask
    
    
    
    '''
    def _get_rawvideo_dec_mang(self, video_path):
        # speed up video decode via decord.
        # video_mask = np.zeros(self.max_frames, dtype=np.long)
        #video_mask = np.zeros((len(starts), self.max_frames), dtype=np.long)
        
        # max_video_length = 0
        #max_video_length = [0] * len(starts)

        # T x 3 x H x W
        #video = np.zeros((len(starts), self.max_frames, 1, 3,
        #                  self.image_resolution[0], self.image_resolution)[1], dtype=np.float)
        
        video = self._read_video(video_path)
        video_mask = torch.ones(video.shape[:2])
        #print(video.shape, video_mask.shape)
        return video.permute((0,1,4,2,3)).unsqueeze(2), video_mask
    
    
    def _read_video_mang(self, video_path):

        video_cap = cv2.VideoCapture(str(video_path))
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        print(fps, video_path)
        hop_size = int(fps / self.feature_framerate)
        
        features = []
        frame_count = 0
        
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break
            
            if frame_count % hop_size == 0:
                feature = []
                frame = self.transform(Image.fromarray(cv2.resize(frame, self.image_resolution)))
                #print('frame :', frame.shape)
                feature.append(frame)
                for i in range(self.max_frames-1):
                    ret, frame = video_cap.read()
                    if not ret:
                        break
                    frame = self.transform(Image.fromarray(cv2.resize(frame, self.image_resolution)))
                    feature.append(frame)
                    frame_count += 1 
                if len(feature) == self.max_frames:
                    features.append(torch.stack(feature, dim=0).permute(0,2,3,1))
            
            frame_count += 1
        #print('len features :', len(features), ' features[0] shape :', features[0].shape)
        # Release resources
        video_cap.release()
        cv2.destroyAllWindows()
        
        # return output
        if len(features) == 0:
            print(video_path)
            return torch.tensor([0])
        #elif len(features) == 1:
        #    return (self.videos[idx], self.dest_videos[idx], torch.stack(features, dim=0))
        else:
            features = torch.stack(features, dim=0)
            return features
    

    def __getitem__(self, idx):
        vdir = self.viddirs[idx]
        video = self.videos[idx]
        
        video_id = vdir.name
        
        video, video_mask = self._get_rawvideo_dec(video)
        
        print(video.shape, video_mask.shape)
        return  video.unsqueeze(1), video_mask.unsqueeze(1)
    '''