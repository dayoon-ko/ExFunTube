from scenedetect import detect, AdaptiveDetector
import torch
import librosa
from pathlib import Path
import os
import json

def scene_detection(video, detector):
    
    try:
        # detect scene boundaries
        scene_list = detect(video, detector)
        
        # to list
        starts, ends = [], []
        for i, scene in enumerate(scene_list):
            s = float(scene[0].get_timecode().split(':')[-1])
            e = float(scene[1].get_timecode().split(':')[-1])
            starts.append(s)
            ends.append(e)
        
        # if None
        if len(starts) == 0:
            starts.append(0.0)
            ends.append(librosa.get_duration(filename=Path(video).parent / 'audio.wav'))
            
        return (starts, ends)
        
    except:
        print(f"Error occurs while downloading {video}")
        return None
        

def segment_with_stt_timestamp(video_dir, starts, ends):
    
    with open(video_dir + '/audio.json') as f:
        stt_result = json.load(f)
        
    new_seg = []
    mark = 0
    
    to_next = False
    # iteration over scene
    for k, (start, end) in enumerate(zip(starts, ends)):
        last_end = start
        sub_seg = {'start':[], 'end':[], 'text':[]}
        
        # segments iteration
        for m, segment in enumerate(stt_result['segments']):
            if m < mark:
                continue
            if segment['end'] < end or to_next or (segment['start'] < end and end - segment['start'] >= segment['end'] - end):
                if segment['start'] - last_end >= 1.0:
                    sub_seg['start'].append(round(last_end,2))
                    sub_seg['end'].append(round(segment['start'],2))
                    sub_seg['text'].append('')
                    last_end = segment['start']
                
                sub_seg['start'].append(round(max(segment['start'], start),2))
                sub_seg['end'].append(round(min(segment['end'], end),2))
                sub_seg['text'].append(segment['text'])
                last_end = segment['end']
                mark += 1
                to_next = False
                continue
            elif segment['start'] < end and end - segment['start'] < segment['end'] - end:
                to_next = True
            if end - last_end >= 1.0:
                sub_seg['start'].append(round(last_end,2))
                sub_seg['end'].append(round(end,2))
                sub_seg['text'].append('')
                last_end = end
            break
        
        if end - last_end >= 1.0:
            sub_seg['start'].append(round(last_end,2))
            sub_seg['end'].append(round(end,2))
            sub_seg['text'].append('')
            last_end = end
            
        if len(sub_seg['start']) >= 1:
            new_seg.append(sub_seg)
        
    return new_seg
