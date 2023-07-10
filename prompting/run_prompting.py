import argparse
from make_prompt import Prompter
from explanation import Explanationer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='../pipeline/videos')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--adaptive_threshold', type=float, default=1.5)
    parser.add_argument('--min_scene_len', type=int, default=15)
    parser.add_argument('--window_width', type=int, default=4)
    parser.add_argument('--min_content_val', type=int, default=6)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    prompter = Prompter(args)
    prompter()
    explanationer = Explanationer(args)
    explanationer()