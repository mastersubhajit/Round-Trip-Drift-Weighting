#!/usr/bin/env python3
"""Evaluate pretrained or trained UNet on held-out prompts using fixed paired seeds.

Creates actual generated PNGs, not synthetic score labels. Requires GPU.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
from data_utils import load_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model',required=True,help='Cached or local SD1.x Diffusers checkpoint')
    p.add_argument('--run-dir',help='Trained run containing unet/; omit to evaluate the pretrained base')
    p.add_argument('--prompts',required=True,help='JSONL: id,prompt,source_concept,target_concept')
    p.add_argument('--out',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55,66,77,88])
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--guidance-scale',type=float,default=7.5)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Image generation requires CUDA GPU')
    out=Path(a.out).resolve()
    if out.exists():raise ValueError('Output directory must not exist (prevents overwriting evaluation results)')
    prompts=load_jsonl(a.prompts)
    if len({r['id'] for r in prompts})!=len(prompts):raise ValueError('Duplicate prompt ID')
    for r in prompts:
        if not {'id','prompt','source_concept','target_concept'} <= set(r):
            raise ValueError('Each prompt requires id,prompt,source_concept,target_concept')
    out.mkdir(parents=True)
    dt=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs={'torch_dtype':dt,'local_files_only':True}
    if a.run_dir:
        run=Path(a.run_dir)
        kwargs['unet']=UNet2DConditionModel.from_pretrained(run/'unet',torch_dtype=dt,local_files_only=True)
        if (run/'text_encoder').is_dir():
            kwargs['text_encoder']=CLIPTextModel.from_pretrained(run/'text_encoder',torch_dtype=dt,local_files_only=True)
    # Safety checker disabled for ALL arms: its NSFW false positives return black images that an
    # automatic judge would score as concept failures. Recorded in eval_config.json.
    kwargs.update(safety_checker=None, requires_safety_checker=False)
    pipe=StableDiffusionPipeline.from_pretrained(a.base_model,**kwargs).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    (out/'eval_config.json').write_text(json.dumps({**vars(a),'prompt_count':len(prompts),
        'safety_checker':None,'scheduler':type(pipe.scheduler).__name__,'dtype':str(dt)},indent=2))
    with torch.inference_mode(), (out/'generations.jsonl.part').open('w',encoding='utf8') as f:
        for row in prompts:
            # One generator per image: each image's initial latent depends only on its own seed,
            # identical to an unbatched call with that seed, so arms stay paired image-by-image.
            gens=[torch.Generator(device='cuda').manual_seed(s) for s in a.seeds]
            images=pipe([row['prompt']]*len(a.seeds),num_inference_steps=a.steps,
                        guidance_scale=a.guidance_scale,generator=gens).images
            for seed,image in zip(a.seeds,images):
                key=sha256(f"{row['id']}|{seed}".encode()).hexdigest()[:24]
                filename=f'{key}.png'
                image.save(out/filename)
                f.write(json.dumps({'image':filename,**row,'seed':seed},ensure_ascii=False)+'\n')
    (out/'generations.jsonl.part').replace(out/'generations.jsonl')  # atomic: present only when complete

if __name__=='__main__':main()
