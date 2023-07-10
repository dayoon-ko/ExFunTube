'''
    Download video, extract audio, and perform speech-to-text and video captioning.
    Finally, gather all the results and store them as a json file.
'''

import argparse
import os
from glob import glob
import json
import torch
import yt_dlp
from multiprocessing import Pool
from pathlib import Path
import whisper
from zero_shot_video_to_text.run import run_videos

########## Download Videos ##########
def filter(info, *, incomplete):
    '''Filter function for sustain only the videos shorter than 30 seconds'''
    duration = info.get('duration')
    if duration and duration>30: 
        return 'The video is too long'
    #category = info.get('categories')[0]
    #if category and (category not in filter_category):
    #    return 'Not funny shorts'

def download(url):
    '''Given url, download the according video to the folder'''
    if 'channel' in url:
        return
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    try:
        error_code = ydl.download(url)
    except Exception as e:
        print(e)


def download_videos(args):
    global ydl_opts
    ydl_opts = {
        'match_filter' : filter,
        'quiet' : 'True',
        'noplaylist' : 'True',
        'outtmpl' : args.root_dir + '/%(id)s/%(title)s.mp4',
        'format' : '136+140/137+140/136+m4a/137+m4a/mp4+140/18/22/mp4+m4a',
    }
    urls = ['https://youtube.com/watch?v=' + i for i in torch.load(args.vid_file) if not os.path.exists(args.root_dir + f'/{i}')]
    with Pool(args.num_workers) as p:
        p.map(download, urls)


########## Extract Audios ##########
def extract_audio(vid_dir):
    # extract audio from given video dir
    try:
        aud_dir = Path(vid_dir).parent / "audio.wav"
        if not os.path.exists(aud_dir):
            os.system(f'yes | ffmpeg -i "{vid_dir}" {aud_dir}')
    except Exception as e:
        print('Error :', vid_dir)

def extract_audios(args):
    # args.root_dir
    # |- {id}
    #    |- {title}.mp4
    #    |-  audio.wav
    videos = glob(os.path.join(args.root_dir, '*/*.mp4'))
    if len(videos) == 0:
        print('Warning : Length of video list is 0.')
    with Pool(args.num_workers) as p:
        p.map(extract_audio, videos)


########## Speech-to-Text ##########
def run_whisper(args):
    videos = [i for i in glob(args.root_dir + '/*') if os.path.isdir(i)]
    # load model
    model = whisper.load_model("large-v2")
    for vid in videos:
        # transcribe audio and store result
        if os.path.exists(vid + '/audio.json'):
            continue
        result = model.transcribe(vid + '/audio.wav')
        with open(vid + '/audio.json', 'w') as f:
            json.dump(result, f, indent=2)
            

########## Video Captioning ##########
def run_vidcap(args):
    videos = glob(args.root_dir + '/*/*.mp4')
    audios = glob(args.root_dir + '/*/audio.json')
    videos_w_stt = []
    for vid, aud in zip(videos, audios):
        with open(aud) as f:
            aud_json = json.load(f)
        if len(aud_json['text']) > 0:
            videos_w_stt.append(vid)
    run_videos(videos_w_stt)


########## Gather ##########
def gather_info(args):
    stts = {}
    jss = [ i for i in glob(args.root_dir + '/*/caption.pt') ]
    
    # check audio and gather video information
    for pth in jss:
        with open(pth[:pth.rfind('/')] + '/audio.json') as f:
            js = json.load(f)
        if js['language'] == 'en':
            info = {'start': [], 'end': [], 'text': []}
            for seg in js['segments']:
                info['start'].append(round(seg['start'], 2))
                info['end'].append(round(seg['end'], 2))
                info['text'].append(seg['text'])
            vcap = torch.load(pth)
            stts[pth.split('/')[-2]] = {'vcap': vcap, 'stt': info}
    # store all the video information
    with open(args.root_dir + '/info.json', 'w') as f:
        json.dump(stts, f, indent=2)


def extract(args):
    # download videos
    print('Download videos...')
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir, exist_ok=True)
    download_videos(args)
    
    # extract audio
    print('Extract audios from videos...')
    extract_audios(args)
    
    # run whisper 
    print('Transcribe audios...')
    run_whisper(args)
    
    # run video captioning
    print('Run video captioning...')
    run_vidcap(args)
    
    # gather results of stt & vcap
    print('Gather outputs')
    gather_info(args)
    print(f"Store speech-to-text & video captioning results of videos at {args.root_dir + '/info.json'}")
    
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='Root folder to store youtube videos', default='videos')
    parser.add_argument('--video_ids', type=str, help='Directory of torch .pt file consisting of youtube ids', default='./video_ids.pt')
    parser.add_argument('--num_workers', help='The number of threads', type=int, default=25)
    args = parser.parse_args()
    return args
        

if __name__ == "__main__":
    args = get_args()
    run(args)
