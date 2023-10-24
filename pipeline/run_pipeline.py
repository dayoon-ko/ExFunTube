import os
import json
import argparse
from extract_info import extract
from prompt_with_json import LanguageModel, FunnyUtteranceFiltering
from calculate_similarity import calculate
from zero_shot_video_to_text.run import *

def pipeline(args):
    # download videos & extract information from videos
    extract(args)
    
    # check api key
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, 'Register your OPENAI_API_KEY before starting'
    lm = LanguageModel(api_key)
    
    # start prompting
    usecase = FunnyUtteranceFiltering(
        lm=lm,
        args=args
    )
    usecase()
    
    # calculate score
    final_result = calculate(args)
    
    # remove filtered videos
    final_videos = {}
    for k, v in final_result.items():
        if v['dv_funny_utterance'] == 'No.':
            continue
        if not v['dv_explanation']:
            final_videos[str(len(final_videos))] = v
            continue
        if v['sentbert'] <= 0.8:
            final_videos[str(len(final_videos))] = v
    
    # save results
    with open(args.prompt_final_result_path, 'w') as f:
        json.dump(final_videos, f, indent=2)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='Root folder to store youtube videos', default='videos')
    parser.add_argument('--num_workers', help='The number of threads', type=int, default=25)
    parser.add_argument('--video_ids', type=str, help='Directory of torch .pt file consisting of video ids', default='./video_ids.pt')
    parser.add_argument('--video_info', type=str, help='Directory of results of speech-to-text and video captioning', default='./videos/info.json')
    parser.add_argument('--prompt_result_path', type=str, help='Directory to store the result of pipeline', default='pipeline_result.json')
    parser.add_argument('--prompt_final_result_path', type=str, help='Directory to store the result of pipeline', default='pipeline_final_result.json')
    
    # zero-shot-video-captioning args
    parser.add_argument("--randomized_prompt", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--db_filter_path", type=str, default=None, help="file to filter db items, e.g karpathy split")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=20)
    parser.add_argument("--cond_text", type=str, default="Image of a")
    parser.add_argument("--token_wise", action="store_true", help="Should we step the optimization at each token gen")
    parser.add_argument("--num_dummy_tokens", type=int, default=2)
    parser.add_argument("--sentence_iterations", type=int, default=25)
    parser.add_argument("--sampling_top_k", type=int, default=1)
    parser.add_argument("--db_start_idx", type=int, default=0)
    parser.add_argument("--db_num_images", type=int, default=0)
    parser.add_argument("--clip_loss_temperature", type=float, default=1)
    parser.add_argument("--clip_scale", type=float, default=1)
    parser.add_argument("--ce_scale", type=float, default=0.8)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--scheduler_type", type=CLIPTextGenerator.SchedType, default='cosine')
    parser.add_argument("--weight_decay_scale", type=float, default=0.03)
    parser.add_argument("--repetition_penalty", type=float, default=2.0, help='How much much to deter deter repeats')
    parser.add_argument("--entity_penalty", type=float, default=2, help='How much to deter CapsLock in middle of sent')
    parser.add_argument("--ending_bonus", type=float, default=2, help='How much to help the sentence to end')
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--pairs_path", type=str, default="")
    parser.add_argument('--data_path', type=str, default='/home/work/Datasets/MSR-VTT/examples/video7157.mp4')
    parser.add_argument('--run_type',
                        default='caption_images',
                        nargs='?',
                        choices=['caption_images', 'caption_videos'])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    pipeline(args)