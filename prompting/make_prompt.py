import argparse
from glob import glob
from pathlib import Path
from segment import scene_detection, segment_with_stt_timestamp
from run_blip import load_model, caption_video
import sys
sys.path.append('InternVideo/Downstream/Video-Text-Retrieval')
from inference import caption_retrieval

class Prompter:
    
    def __init__(self, args):
        self.device = args.device
        self.adaptive_threshold = args.adaptive_threshold #1.5
        self.min_scene_len = args.min_scene_len #15
        self.window_width = args.window_width #4
        self.min_content_val = args.min_content_val #6
        self.root_dir = args.root_dir
        self.video_dirs = [Path(i) for i in sorted(glob(args.root_dir + '/*'))]
        self.videos = [glob(str(i / '*.mp4')) for i in self.video_dirs]
        self.caption_filename = 'caption_corpus.pt'
        self.segmentation_filename = 'segments.json'
    
    #####################
    # 1. Segment videos #
    #####################
    def _segment_videos(self):
        Detector = AdaptiveDetector(adaptive_threshold=self.adaptive_threshold,
                                    min_scene_len=self.min_scene_len,
                                    window_width=self.window_width,
                                    min_content_val=self.min_content_val)
        for  video_dir, video in zip(self.video_dirs, self.videos):
            starts, ends = scene_detection(video, Detector)
            result = segment_with_stt_timestamp(str(video_dir), starts, ends)
            with open(video_dir / self.segmentation_filename, 'w') as f:
                json.dump(result, f, indent=2)
    
    
    #########################
    # 2. Visual Description #
    #########################
    def _generate_caption_corpus(self):
        model, vis_processors = load_model(self.device)
        for video_dir, video in zip(self.video_dirs, self.videos):
            captions = caption_video(video)
            torch.save(captions, video_dir / self.caption_filename)
    
    def _select_frame_captions(self):
        caption_retrieval(self.root_dir, self.caption_filename, self.segmentation_filename)

    
    #############
    # 3. Speech #
    #############
    def _speaker_diarization(self):
        # To Do
        return None
    
    
    def __call__(self):
        self._segment_videos()
        self._generate_caption_corpus()
        self._select_frame_captions()
        self._speaker_diarization()
        