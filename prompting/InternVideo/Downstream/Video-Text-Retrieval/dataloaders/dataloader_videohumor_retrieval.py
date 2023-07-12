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

    
class VideoHumor_DataLoader(Dataset):
    """MSRVTT dataset loader."""
    def __init__(
            self,
            video_dir,
            caption_filename,
            segmentation_filename,
            tokenizer,
            max_words=30,
            feature_framerate=5.0,
            max_frames=200,
            image_resolution=224
    ):  
        self.viddirs = sorted([i for i in glob(video_dir + '/*') if os.path.isdir(i)], key=lambda x: x.lower())
        self.videos = [glob(f'{p}/*.mp4')[0] for p in self.viddirs]
        self.texts = [f'{p}/{caption_filename}' for p in self.viddirs]
        self.segments = [f'{p}/{segmentation_filename}' for p in self.viddirs]
        
        self.feature_framerate = feature_framerate
        self.sen_sample_rate = 5
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.image_resolution = image_resolution

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

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

    def _get_text(self, sentences_set, starts=None, ends=None, t_stride=None):
        output = [[],[],[]]
        total_sentences = []
        for s, e in zip(starts, ends):
            if s >= e:
                continue

            sentences = []
            for i in range(s, e, t_stride):
                sentences.extend(sentences_set[int(float(i)/t_stride)])
            
            k=len(sentences)
            pairs_text = np.zeros((k, self.max_words), dtype=np.long)
            pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
            pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
            
            for i, sentence in enumerate(sentences):
                
                words = self.tokenizer.tokenize(sentence)

                words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
                total_length_with_CLS = self.max_words - 1
                if len(words) > total_length_with_CLS:
                    words = words[:total_length_with_CLS]
                words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

                input_ids = self.tokenizer.convert_tokens_to_ids(words)
                input_mask = [1] * len(input_ids)
                segment_ids = [0] * len(input_ids)
                while len(input_ids) < self.max_words:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                assert len(input_ids) == self.max_words
                assert len(input_mask) == self.max_words
                assert len(segment_ids) == self.max_words

                pairs_text[i] = np.array(input_ids)
                pairs_mask[i] = np.array(input_mask)
                pairs_segment[i] = np.array(segment_ids)
            
            output[0].append(torch.from_numpy(pairs_text))
            output[1].append(torch.from_numpy(pairs_mask))
            output[2].append(torch.from_numpy(pairs_segment))

            total_sentences.append(sentences)
            
        return output[0], output[1], output[2], total_sentences

    def _get_rawvideo_dec(self, vreader, starts=None, ends=None, t_stride=None):
        # speed up video decode via decord.
        # video_mask = np.zeros(self.max_frames, dtype=np.long)
        video_mask = np.zeros((len(starts), self.max_frames), dtype=np.long)
        
        # max_video_length = 0
        max_video_length = [0] * len(starts)

        # T x 3 x H x W
        video = np.zeros((len(starts), self.max_frames, 1, 3,
                          self.image_resolution, self.image_resolution), dtype=np.float)

        for i, (f_start, f_end) in enumerate(zip(starts, ends)):
            #try:
            if f_end >= f_start:
                # T x 3 x H x W
                all_pos = list(range(f_start, f_end + 1, t_stride))
                if len(all_pos) > self.max_frames:
                    sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=self.max_frames, dtype=int)]
                else:
                    sample_pos = all_pos
                patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).numpy()]
                patch_images = torch.stack([self.transform(img) for img in patch_images])
                
                patch_images = patch_images.unsqueeze(1)
                
                slice_len = patch_images.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = patch_images
            else:
                print("video path error.")#.format(video_path))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return torch.from_numpy(video), torch.from_numpy(video_mask)
    
    def _read_video(self, video):
        # open
        video = cv2.VideoCapture(video)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if not video.isOpened():
            print(f'Video error : {video}')
        
        # read
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
        video.release()
        cv2.destroyAllWindows()
        
        return fps, video.get(cv2.CAP_PROP_FPS), frames    
    
    
    def _get_tsmps(self, starts, ends, n_frames, fps):
        starts = [int(i * fps) for i in starts]
        ends = [int(i * fps) for i in ends]
        n_starts, n_ends = [], []
        for s, e in zip(starts, ends):
            if s < n_frames:
                n_starts.append(s)
                if e < n_frames:
                    n_ends.append(e)
                else:
                    n_ends.append(n_frames-1)
                    break
        return n_starts, n_ends


    def __getitem__(self, idx):
        vdir = self.viddirs[idx]
        video = self.videos[idx]
        sentences = torch.load(self.texts[idx])
        with open(self.segments[idx]) as f:
            segments = json.load(f)

        video_id = vdir.split('/')[-1]
        print(video_id)
        vreader = VideoReader(video, ctx=cpu(0))
        fps = vreader.get_avg_fps()
        len_frames = len(vreader)
        duration = float(len_frames) / fps
        dur = librosa.get_duration(filename=f'{vdir}/audio.wav')
        
        out = []
        for seg in segments:
            starts, ends = self._get_tsmps(seg['start'], seg['end'], len_frames, fps)
            pairs_text, pairs_mask, pairs_segment, sentences_out = self._get_text(sentences, starts, ends, int(round(float(fps) / float(self.sen_sample_rate))))
            video, video_mask = self._get_rawvideo_dec(vreader, starts, ends, int(round(float(fps) / float(self.feature_framerate))))
            out.append([video_id, pairs_text, pairs_mask, pairs_segment, video, video_mask, sentences_out]) #, s, starts, ends
        return out
