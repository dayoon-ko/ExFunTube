# ExFunTube
The source code of ExFunTube

## Usage

### To run filter pipeline:

```bash
$ cd pipeline
$ conda create --name {env_name}
$ conda env create --file environment.yaml
$ python run_pipeline.py --video_ids {video_id_file_name}
```

### To run prompting:

```bash
$ cd prompting
$ conda create --name {env_name}
$ conda env create --file environment.yaml
$ python run_prompting.py
```
