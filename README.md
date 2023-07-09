# ExFunTube
The source code of ExFunTube
# Official Implementation of [Zero-Shot Video Captioning with Evolving Pseudo-Tokens](https://arxiv.org/abs/2207.11100)

## Usage

### To run filter pipeline:

```bash
$ cd pipeline
$ python run_pipeline.py --video_ids {video_id_file_name}
```

### To run prompting:

```bash
$ python run.py 
--token_wise --randomized_prompt
--run_type caption_images
--data_path examples/example_image.jpg
```
