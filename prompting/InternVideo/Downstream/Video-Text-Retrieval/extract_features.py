from inference_seg import get_args, init_model, set_seed_logger, init_device
from glob import glob
from dataloaders.dataloader_feature_extraction import VideoHumorDataset

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


if __name__=="__main__":
    
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.rank)
    
    model = init_model(args, device, n_gpu, args.rank)
    dataset = VideoHumorDataset(num=args.num)
    
    def collate_fn(batch):
        return batch[0]
    
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=False, collate_fn=collate_fn)
    
    for batch in tqdm(dataloader):
        output = []
        vid, video, video_mask = batch[0], batch[1].to(device), batch[2].to(device)
        print(video.shape)
        chunk = 3
        for i in range(0, video.shape[0], chunk):
            result = model.get_visual_output(video[i:i+chunk], video_mask[i:i+chunk])
            output.extend(result.cpu().detach().numpy())
        
        output = np.stack(output).squeeze(1)
        print(output.shape)
        torch.save(output, f'/gallery_louvre/dayoon.ko/dataset/videohumor_maf/video_feats_iv/{vid}.pt')