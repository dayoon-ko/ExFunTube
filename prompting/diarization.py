import json
import openai
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import os
import torch
from sacrebleu import BLEU
from torchmetrics import BLEUScore
from collections import OrderedDict
import signal
import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from glob import glob
from tqdm import tqdm


class Diarization:
    def __init__(self, lm, args):
        self.lm = lm
        self.metas = [i for i in sorted(glob(str(Path(args.meta))), key=lambda x:x.lower()) if os.path.exists(i)]
        self.vids = [i.split('/')[-2] for i in self.metas]
        self.output_pth = Path(args.out_pth) 
        self.js = {}
        if args.init_pth:
            with open(args.init_pth) as f:
                js = json.load(f)
            for i in self.vids:
                if i in js:
                    self.js[i] = js[i]
            del js
                
        
    def _get_transcript(self, meta):
        transcript = ''
        num = 0
        for i, scene in enumerate(meta):
            for k, utter in enumerate(scene['text']):
                if len(utter) > 0:
                    transcript += f'{utter}' + '\n'
                    num += 1
        transcript = 'Transcript: ' + transcript
        return transcript, num
    

    def _make_num_speaker_prompt(self, meta):
        transcript, num = self._get_transcript(meta)
        if num == 0 or num == 1:
            return transcript, None
        prompt1 = f'Based on the context of the given transcript, how many speakers are there? Please provide only the number.\n'\
                f'{transcript}'\
                f'What is the most likely number of speakers in the given transcript? Please provide only the most likely number and no other explanation.'
        prompt1 = [
                    {"role" : "system", "content" : "You are a helpful assistant."},
                    {"role" : "user", "content" : prompt1}
                ]
        return transcript, prompt1, num
    

    def _make_diarization_prompt(self, prompt1, res1):
        prompt2 = f'Based on the context of the given transcript and your estimated number of speakers, please assign speakers to the transcript.'
        prompt2 = [
                    {"role" : "system", "content" : "You are a helpful assistant."},
                    {"role" : "user", "content" : prompt1[-1]['content']},
                    {"role" : "user", "content" : res1},
                    {"role" : "user", "content" : prompt2}
                ]
        res2 = self._run_lm(prompt2)
        return prompt2, res2
    

    def _make_diarization_prompt_sequentially(self, transcript, prompt1, res1):
        output = ['Speaker 1'] 
        if len(transcript) > 1:
            prompt2 = [
                    {"role" : "system", "content" : "You are a helpful assistant."},
                    {"role" : "user", "content" : prompt1},
                    {"role" : "system", "content" : res1},
                    {"role" : "user", "content" : f"Please let me know the most likely speaker number given a sentence in the transcript. "\
                                                    f"Example) Sentence: {transcript[0]} Speaker Number : Speaker 1. Sentence: {transcript[1]}"},
                    ]
            res2 = self._run_lm(prompt2)
            output.append(res2)
            prompt2.append({"role": "system", "content": res2})
        for s in transcript[2:]:
            prompt2.append({"role" : "user", "content" : f'Sentence: {s}'})
            res2 = self._run_lm(prompt2)
            output.append(res2)
            prompt2.append({"role": "system", "content": res2})
        return prompt2, output 


    def _run_lm(self, prompt):
        return self.lm(prompt)


    def _is_not_dialogue(self, res):
        if 'one' in res or '1' in res or '0' in res:
            return True
        return False


    def _post_processing(self, meta, num):
        if num == 1:
            diar = []
            for txt in meta['text']:
                if len(txt) > 0:
                    diar.append("")
                else:
                    diar.append("1")
            meta['diar'] = diar
            return meta
        else:
            
    
    def __call__(self):
        
        for i, aud in tqdm(enumerate(self.metas), total= len(self.metas)):
            
            with open(aud) as f:
                stt = json.load(f)
            aud_id = aud.split('/')[-2]
            
            transcript, prompt1, num = self._make_num_speaker_prompt(stt)
            
            # if no transcript
            if num == 0: 
                continue
            
            # if only one utterance
            if num == 1:
                meta = self._post_processing(meta, 1)
                
            # if there are more than two utterances
            else:
                res1 = self._run_lm(prompt1)
                
                # if speakers < 2
                if self._is_not_dialogue(res.lower()):
                    meta = self._post_processing(meta, 1)
                    
                # speakers >= 2
                else:
                    prompt2 = self._make_diarization_prompt(prompt1, res1)
                    res2 = self._run_lm(prompt2)
        
        
        for i, aud in tqdm(enumerate(self.metas), total=len(self.metas)):
            aud_id = aud.split('/')[-2]
            transcript, done = self._get_transcript_list(self.output[aud_id])
            
            if 'res3' in self.js[aud_id]:
                result = postprocess_dialogue_recur(transcript, self.js[aud_id]['res3'], self.output[aud_id])
                self.output[aud_id] = result
                
            elif 'res2' in self.js[aud_id] and not done:
                #try:
                prompt3, res3 = self._make_diarization_prompt_sequentially(transcript, self.js[aud_id]['prompt1'], self.js[aud_id]['res1'])
                self.js[aud_id]['prompt3'] = prompt3
                self.js[aud_id]['res3'] = res3
                result = postprocess_dialogue_recur(transcript, res3, self.output[aud_id])
                self.output[aud_id] = result
            

class LanguageModel:
    def __init__(self, key):
        openai.api_key = key

    def __call__(self, prompt, engine="text-davinci-003", temperature=0.1) -> str:  
        while True:
            try:
                res = openai.ChatCompletion.create(
                    model = "gpt-3.5-turbo",
                    messages=prompt,
                    temperature=0
                )
                return res['choices'][0]['message']['content']
            except Exception as e:
                print(e)
                continue
    
            
def alarm_handler(signum, frame):
    raise Exception('timeout')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', default='/gallery_louvre/dayoon.ko/research/mmvh/3Eem2ZVliB4/segments_.json')
    parser.add_argument('--out_pth', default='/gallery_louvre/dayoon.ko/research/mmvh/videohumor/diar_output')
    args = parser.parse_args()

    # alarm
    signal.signal(signal.SIGALRM, alarm_handler)
    
    # output path
    current_date = datetime.today().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    out_pth = Path(args.out_pth) / current_date / current_time
    if not out_pth.exists():
        out_pth.mkdir(parents=True)
    args.out_pth = out_pth
    
    api_key = os.getenv("OPENAI_API_KEY")
    gpt3_api = LanguageModel(api_key)
    usecase = Diarization(
        lm=gpt3_api,
        args=args,
    )
    usecase()
        
