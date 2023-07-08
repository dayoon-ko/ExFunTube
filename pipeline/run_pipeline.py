import os
import json
import argparse
from extract_info import extract
from prompt_with_json import LanguageModel, FunnyUtteranceFiltering
from calculate_similarity import calculate

def pipeline(args):
    # Download videos & Extract information from videos
    extract(args)
    
    # Check api key
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, 'Register your OPENAI_API_KEY before starting'
    lm = LanguageModel(api_key)
    
    # Start prompting
    usecase = FunnyUtteranceFiltering(
        lm=lm,
        args=args
    )
    usecase()
    
    # Calculate score
    final_result = calculate(args)
    
    # Remove filtered videos
    final_videos = {}
    for k, v in final_result.items():
        if v['dv_funny_utterance'] == 'No.':
            continue
        if not v['dv_explanation']:
            final_videos[str(len(final_videos))] = v
            continue
        if v['sentbert'] <= 0.8:
            final_videos[str(len(final_videos))] = v
    with open(args.prompt_final_result_path, 'w') as f:
        json.dump(final_videos, f, indent=2)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='Root folder to store youtube videos', default='videos')
    parser.add_argument('--vid_file', type=str, help='Directory of torch .pt file consisting of youtube ids', default='./video_ids.pt')
    parser.add_argument('--num_workers', help='The number of threads', type=int, default=25)
    parser.add_argument('--video_ids', type=str, help='Directory of torch .pt file consisting of video ids', default='./video_ids.pt')
    parser.add_argument('--video_info', type=str, help='Directory of results of speech-to-text and video captioning', default='./videos/info.json')
    parser.add_argument('--prompt_result_path', type=str, help='Directory to store the result of pipeline', default='pipeline_result.json')
    parser.add_argument('--prompt_final_result_path', type=str, help='Directory to store the result of pipeline', default='pipeline_final_result.json')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    pipeline(args)