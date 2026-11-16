#!/usr/bin/env python3
"""Noisy Upscaling (Hönig, Rando, Carlini, Tramèr, "Adversarial Perturbations Cannot Reliably Protect Artists
From Generative AI", ICLR 2025) - public code https://github.com/ethz-spylab/robust-style-mimicry @ b4bf72b (MIT).

Step 1 is the authors' unmodified `noise.py` (called by run_arm.py). This file is step 2, a faithful port of their
`upscale.py`: StableDiffusionUpscalePipeline (stabilityai/stable-diffusion-x4-upscaler), prompt = the image's own
caption, noise_level=320, noise_level_sub=320, batch 1, pipeline defaults otherwise (75 steps, guidance 9.0, fp32).
`noise_level_sub` comes from their diffusers fork (roberthoenig/diffusers @5048e9b), whose ONLY change is
    image = low_res_scheduler.add_noise(image, noise, noise_level - noise_level_sub)
and it is reproduced here by wrapping that call, so stock diffusers 0.33 gives the same computation.
Differences, recorded in purify_meta.json: model loaded from a local pinned copy (offline nodes); output is resized
2048->512 with the same bicubic resize the trainer applies anyway (saves ~19 GB; training input identical);
--rank/--world shard the work over GPUs; finished files are skipped on rerun.

--helper (one GPU upscales ~38 images/h, so 2,400 images do not fit in a 24 h job):
extra processes (e.g. a job on another node) fill the SAME --out-dir while the arm's own
forward-order process keeps running. A helper walks the list in REVERSE, claims each file with an O_EXCL lock file
(several helpers never duplicate work), writes to a temp file in <out-dir>.helper/ and os.replace()s it in, so the forward process
and the trainer can only ever see complete images. Computation per image is unchanged (same pipeline, settings, seed
handling: none, as in the authors' code); only which process computes an image differs.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path


def run(names, dst: Path, upscale, helper=False):
    """upscale(name) -> PIL image (already 512-resized). Returns the number of images this process wrote.
    Helper claims and temp files live in the sibling `<dst>.helper/` (same filesystem, so os.replace is atomic) and
    never inside dst, which the arm renames to train/ when its own pass ends; a helper then stops."""
    done, side = 0, dst.with_name(dst.name + '.helper')
    if helper: side.mkdir(exist_ok=True)
    for name in (reversed(names) if helper else names):
        if (dst / name).exists(): continue
        if not helper:
            upscale(name).save(dst / name); done += 1; continue
        if not dst.exists(): break
        try: os.close(os.open(side / f'{name}.claim', os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError: continue   # another helper has it
        if (dst / name).exists(): continue
        tmp = side / f'{name}.tmp.png'
        upscale(name).save(tmp)
        try: os.replace(tmp, dst / name)
        except FileNotFoundError: break   # dst was renamed meanwhile: the arm finished this image itself
        done += 1
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--in-dir', required=True, help='noise.py output: PNGs + metadata.csv (file_name,text)')
    p.add_argument('--out-dir', required=True); p.add_argument('--model', required=True)
    p.add_argument('--noise-level', type=int, default=320); p.add_argument('--noise-level-sub', type=int, default=320)
    p.add_argument('--rank', type=int, default=0); p.add_argument('--world', type=int, default=1)
    p.add_argument('--dtype', choices=['fp32', 'fp16'], default='fp32', help='authors used fp32 (default)')
    p.add_argument('--helper', type=int, default=None, metavar='ID', help='helper process ID: reverse order + claims + atomic writes (see docstring)')
    a = p.parse_args()
    import csv, torch
    from PIL import Image
    from diffusers import StableDiffusionUpscalePipeline
    src, dst = Path(a.in_dir), Path(a.out_dir)
    if a.helper is not None and not dst.is_dir(): raise SystemExit(f'helper: {dst} does not exist (arm finished or not started)')
    dst.mkdir(parents=True, exist_ok=True)
    texts = {r['file_name']: r['text'] for r in csv.DictReader((src / 'metadata.csv').open())}
    names = sorted(n for n in texts if n.endswith(('.png', '.jpg')))[a.rank::a.world]
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        a.model, local_files_only=True, torch_dtype=torch.float16 if a.dtype == 'fp16' else torch.float32).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    orig_add = pipe.low_res_scheduler.add_noise
    sub = a.noise_level_sub
    pipe.low_res_scheduler.add_noise = lambda x, n, t: orig_add(x, n, t - sub)   # == fork's one-line change
    t0 = time.time()

    def upscale(name):
        with Image.open(src / name) as im:
            img = im.convert('RGB')
        out = pipe(prompt=[texts[name]], image=[img], noise_level=a.noise_level).images[0]
        w, h = out.size; s = 512 / min(w, h)
        return out.resize((round(w * s), round(h * s)), Image.Resampling.BICUBIC)

    done = run(names, dst, upscale, a.helper is not None)
    if a.helper is not None and not dst.exists(): return   # the arm already finished and renamed the folder
    (dst / (f'purify_meta_helper{a.helper}.json' if a.helper is not None else f'purify_meta_rank{a.rank}.json')).write_text(json.dumps(dict(
        method='noisy_upscaling (Hoenig et al. 2025)', code='ethz-spylab/robust-style-mimicry@b4bf72b upscale.py, ported',
        fork_change='roberthoenig/diffusers@5048e9b add_noise(noise_level - noise_level_sub)',
        noise_level=a.noise_level, noise_level_sub=sub, dtype=a.dtype, steps='pipeline default (75)',
        guidance='pipeline default (9.0)', images=done, seconds=time.time() - t0,
        seconds_per_image=(time.time() - t0) / max(done, 1)), indent=2))
    print(json.dumps(dict(rank=a.rank, images=done, seconds=time.time() - t0)))


if __name__ == '__main__':
    main()
