
import argparse
import torch
import librosa
import numpy as np
from torch import autocast
from contextlib import nullcontext
import json
from glob import glob

from models.MobileNetV3 import get_model as get_mobilenet, get_ensemble_model
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels

def get_args():
    parser = argparse.ArgumentParser(description='Example of parser. ')
    # model name decides, which pre-trained model is loaded
    parser.add_argument('--model_name', type=str, default='mn10_as')
    parser.add_argument('--strides', nargs=4, default=[2, 2, 2, 2], type=int)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--demo', action='store_true', default=False)
    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_mels', type=int, default=128)

    # overwrite 'model_name' by 'ensemble_model' to evaluate an ensemble
    parser.add_argument('--ensemble', nargs='+', default=["mn40_as_ext"]) #, "mn40_as", "mn40_as_no_im_pre"])

    args = parser.parse_args() 
    return args



def run(model, mel, sample_rate, audio_path, device):

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)

    with torch.no_grad(), autocast(device_type=device.type):
        spec = mel(waveform)
        preds, features = model(spec.unsqueeze(0))
    preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()


    sorted_indexes = np.argsort(preds)[::-1]
    result = [ (labels[i], float(preds[i])) for i in sorted_indexes[:15]]
    
    return result



def audio_tagging(root_dir, audtag_filename): 
    """
    Running Inference on an audio clip.
    """
    
    args = get_args()
    
    model_name = args.model_name
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    n_mels = args.n_mels

    # load pre-trained model
    if len(args.ensemble) > 0:
        model = get_ensemble_model(args.ensemble)
    else:
        model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name, strides=args.strides,
                              head_type=args.head_type)
    model.to(device)
    model.eval()

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size)
    mel.to(device)
    mel.eval()
    
    audios = glob(root_dir + '/*/audio.wav')
    for audio in audios:
        result = run(model, mel, sample_rate, audio, device)
        torch.save(result, audio[:audio.rfind('/')] + '/' + audtag_filename)


if __name__ == "__main__":
    audio_tagging()