#!/usr/bin/env python3
"""Full-parameter SD1.x UNet training on real, audited image-caption records.

This is a reproducibility harness for three research arms, NOT proof of a new
Nightshade defense. Frozen VAE by design; optional all-parameter text-encoder
training. `--init-unet random` initializes only UNet, not the VAE or text encoder.
Requires a CUDA GPU and accelerate/diffusers/transformers.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import math
from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed, GradientAccumulationPlugin
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from data_utils import ImageMetadataDataset, collate, anchor_negative_texts, EpochPermutationSampler
from preflight import inspect_inputs
from training_objectives import per_example_denoising_loss, contrastive_denoising_margin


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Local Diffusers SD1.x directory staged before the GPU job')
    p.add_argument('--train-dir', required=True, help='Directory with REAL RGB files and metadata.jsonl')
    p.add_argument('--trusted-dir', help='Independent rights-approved anchor ImageFolder')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--method', choices=['standard', 'anchor_replay', 'semantic_margin', 'vaccine'], default='standard',
                   help='vaccine = anchor replay + ALL-PAIRS rival-caption margin over --vocabulary (arm V)')
    p.add_argument('--vocabulary', nargs='+', help='concept words for the vaccine margin (default: negative-map keys)')
    p.add_argument('--latent-cache', help='torch file {record_id: unscaled VAE latent [4,H/8,W/8]} used INSTEAD of '
                   'encoding the training pixels (arm R, DCLR latents from scripts/dclr_latents.py)')
    p.add_argument('--negative-map', help='JSON mapping trusted concept to rival concept')
    p.add_argument('--sample-weights', help='JSON {"weights": {record_id: w in [0,1]}} for per-example '
                   'denoising-loss weights (RTDW candidate). Loss = sum(w*l)/(B*mean_w_dataset).')
    p.add_argument('--init-unet', choices=['pretrained', 'random'], default='pretrained')
    p.add_argument('--train-text-encoder', action='store_true', help='Train ALL CLIP text-encoder parameters too (memory intensive)')
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--anchor-batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--max-steps', type=int, default=2000, help='Number of optimizer steps (not examples)')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--text-lr', type=float, default=1e-6)
    p.add_argument('--anchor-weight', type=float, default=1.0)
    p.add_argument('--margin-weight', type=float, default=0.2)
    p.add_argument('--margin', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--mixed-precision', choices=['no', 'fp16', 'bf16'], default='bf16')
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--optimizer', choices=['adamw', 'adamw8bit'], default='adamw8bit')
    p.add_argument('--checkpoint-every', type=int, default=500)
    p.add_argument('--resume-from', help='Accelerate checkpoint-N directory')
    p.add_argument('--deterministic', action='store_true', help='Use deterministic CUDA algorithms and math attention for reproducibility checks')
    p.add_argument('--dry-run', action='store_true', help='Audit local inputs without loading models or training')
    p.add_argument('--log-every', type=int, default=25)
    return p.parse_args()


def configure_determinism(enabled):
    if enabled:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output already contains files: {path}; use a fresh directory, including for resumed runs')
    path.mkdir(parents=True, exist_ok=True)


def accumulation_plugin(steps):
    # Nested prepared anchor loaders must never decide optimizer boundaries.
    # Carry partial accumulation across epochs so every update has exactly N batches.
    return GradientAccumulationPlugin(num_steps=steps, sync_with_dataloader=False)


def optimizer_update_completed(sync_gradients, step_was_skipped):
    return sync_gradients and not step_was_skipped


def set_loader_epoch(loader, sampler, epoch):
    # Accelerate's distributed BatchSamplerShard does not expose its sampler.
    sampler.set_epoch(epoch)
    loader.set_epoch(epoch)


def require_nonempty_loaders(train_loader, anchor_loader):
    if len(train_loader) == 0:
        raise ValueError('Training loader empty after DDP sharding; increase data or reduce batch size')
    if anchor_loader is not None and len(anchor_loader) == 0:
        raise ValueError('trusted loader empty after DDP sharding; increase trusted data or reduce anchor batch size')


def normalize_cursor(epoch, batches_in_epoch, loader_length):
    if epoch < 0 or not 0 <= batches_in_epoch <= loader_length:
        raise ValueError('Invalid checkpoint data cursor')
    if batches_in_epoch == loader_length:
        return epoch + 1, 0
    return epoch, batches_in_epoch


def check_resume_contract(saved, current):
    for key in sorted(set(saved) | set(current)):
        if saved.get(key) != current.get(key):
            raise ValueError(f'Resume contract mismatch for {key}: {saved.get(key)!r} != {current.get(key)!r}')


def resume_contract(args, inputs, world_size):
    mutable = {'max_steps', 'resume_from', 'output_dir', 'log_every', 'checkpoint_every', 'dry_run', 'num_workers'}
    config = {k: v for k, v in vars(args).items() if k not in mutable}
    # Content identities allow moving identical assets to a different mount.
    for key in ['model', 'train_dir', 'trusted_dir', 'negative_map', 'sample_weights', 'latent_cache']:
        config.pop(key, None)
    config.update(world_size=world_size, train_fingerprint=inputs['train']['fingerprint'],
                  trusted_fingerprint=inputs['trusted']['fingerprint'] if inputs['trusted'] else None,
                  model_fingerprint=inputs['model_fingerprint'],
                  negative_map_sha256=sha256(Path(args.negative_map).read_bytes()).hexdigest() if args.negative_map else None,
                  sample_weights_sha256=sha256(Path(args.sample_weights).read_bytes()).hexdigest() if getattr(args, 'sample_weights', None) else None,
                  latent_cache_sha256=sha256(Path(args.latent_cache).read_bytes()).hexdigest() if getattr(args, 'latent_cache', None) else None)
    return config


def get_target(scheduler, latents, eps, timesteps):
    typ = scheduler.config.prediction_type
    if typ == 'epsilon':
        return eps
    if typ == 'v_prediction':
        return scheduler.get_velocity(latents, eps, timesteps)
    raise ValueError(f'Unsupported scheduler prediction type: {typ}')


def encode_pixels(vae, scheduler, pixels, dtype, latents=None):
    with torch.no_grad():
        if latents is None:
            latents = vae.encode(pixels.to(dtype=dtype)).latent_dist.sample()
        else:   # pre-computed (e.g. decoder-consistent) latents replace the encoder output
            latents = latents.to(device=pixels.device, dtype=dtype)
        latents = latents * vae.config.scaling_factor
        eps = torch.randn_like(latents)
        t = torch.randint(0, scheduler.config.num_train_timesteps,
                          (latents.shape[0],), device=latents.device, dtype=torch.long)
        noised = scheduler.add_noise(latents, eps, t)
        target = get_target(scheduler, latents, eps, t)
    return noised, t, target


def prompt_embeddings(encoder, tokenizer, captions, device, train_text_encoder):
    token_ids = tokenizer(
        captions, padding='max_length', truncation=True,
        max_length=tokenizer.model_max_length, return_tensors='pt'
    ).input_ids.to(device)
    if train_text_encoder:
        return encoder(token_ids)[0]
    with torch.no_grad():
        return encoder(token_ids)[0]


def training_loss(unet, scheduler, vae, encoder, tokenizer, pixels, captions, dtype, train_text_encoder,
                  weights=None, latents=None):
    zt, timesteps, target = encode_pixels(vae, scheduler, pixels, dtype, latents)
    hidden = prompt_embeddings(encoder, tokenizer, captions, pixels.device, train_text_encoder)
    prediction = unet(zt, timesteps, hidden).sample
    per_example = per_example_denoising_loss(prediction, target)
    if weights is None:
        return per_example.mean()
    # weights are pre-normalised by the dataset mean weight, so the expected loss scale (and
    # hence effective learning rate) matches the unweighted arms.
    return (per_example * weights.to(per_example)).mean()


def load_sample_weights(path, train_dir):
    spec = json.loads(Path(path).read_text())
    w = {str(k): float(v) for k, v in spec['weights'].items()}
    ids = [json.loads(l)['id'] for l in (Path(train_dir) / 'metadata.jsonl').read_text().splitlines() if l.strip()]
    missing = [i for i in ids if i not in w]
    if missing:
        raise ValueError(f'{len(missing)} training records lack a sample weight, e.g. {missing[:3]}')
    if any(not (0.0 <= w[i] <= 1.0) or not math.isfinite(w[i]) for i in ids):
        raise ValueError('sample weights must be finite and in [0,1]')
    mean = sum(w[i] for i in ids) / len(ids)
    if mean <= 0:
        raise ValueError('all sample weights are zero')
    return {i: w[i] / mean for i in ids}, mean


def trusted_margin_losses(unet, scheduler, vae, encoder, tokenizer, pixels,
                          captions, negative_captions, dtype, train_text_encoder, margin):
    """Matched latents/timestep/noise; rivals differ only in semantic caption.

    Lower loss for the correct caption is a SURROGATE, not generation accuracy.
    """
    zt, timesteps, target = encode_pixels(vae, scheduler, pixels, dtype)
    pos_emb = prompt_embeddings(encoder, tokenizer, captions, pixels.device, train_text_encoder)
    neg_emb = prompt_embeddings(encoder, tokenizer, negative_captions, pixels.device, train_text_encoder)
    pos_loss = per_example_denoising_loss(unet(zt, timesteps, pos_emb).sample, target)
    neg_loss = per_example_denoising_loss(unet(zt, timesteps, neg_emb).sample, target)
    semantic_margin = contrastive_denoising_margin(pos_loss, neg_loss, margin)
    return pos_loss.mean(), semantic_margin


def rival_captions(text, concept, vocabulary):
    """All single-word concept swaps of a trusted caption (fails closed if the concept word is absent)."""
    from data_utils import concept_negative_caption
    return [concept_negative_caption(text, concept, c) for c in vocabulary if c != concept]


def vaccine_margin_losses(unet, scheduler, vae, encoder, tokenizer, pixels, captions, concepts, vocabulary,
                          dtype, train_text_encoder, margin):
    """Arm V: for each trusted image x of concept c and EVERY rival concept c' in the vocabulary,
    softplus(m - (l(x, c') - l(x, c))) with shared latent/timestep/noise. For trusted images of the attacker's
    target concept this includes the source caption, the direction that opposes the poison gradient
    without knowing which concept is attacked."""
    zt, timesteps, target = encode_pixels(vae, scheduler, pixels, dtype)
    pos_emb = prompt_embeddings(encoder, tokenizer, captions, pixels.device, train_text_encoder)
    pos_loss = per_example_denoising_loss(unet(zt, timesteps, pos_emb).sample, target)
    unknown = sorted(set(concepts) - set(vocabulary))
    if unknown:
        raise ValueError(f'trusted concepts {unknown} are not in the vaccine vocabulary')
    rivals = [rival_captions(t, c, vocabulary) for t, c in zip(captions, concepts)]
    n_rival = len(rivals[0])
    if n_rival == 0 or any(len(r) != n_rival for r in rivals):
        raise ValueError('every trusted concept must be in the vocabulary, with >= 1 rival')
    margins = []
    for j in range(n_rival):
        emb = prompt_embeddings(encoder, tokenizer, [r[j] for r in rivals], pixels.device, train_text_encoder)
        neg_loss = per_example_denoising_loss(unet(zt, timesteps, emb).sample, target)
        margins.append(contrastive_denoising_margin(pos_loss, neg_loss, margin))
    return pos_loss.mean(), torch.stack(margins).mean(), n_rival


def load_latent_cache(path, train_dir, resolution):
    cache = torch.load(path, map_location='cpu', weights_only=True)
    ids = [json.loads(l)['id'] for l in (Path(train_dir) / 'metadata.jsonl').read_text().splitlines() if l.strip()]
    missing = [i for i in ids if i not in cache]
    if missing:
        raise ValueError(f'latent cache lacks {len(missing)} training records, e.g. {missing[:3]}')
    shape = (4, resolution // 8, resolution // 8)
    for i in ids:
        if tuple(cache[i].shape) != shape or not torch.isfinite(cache[i].float()).all():
            raise ValueError(f'latent for {i} has shape {tuple(cache[i].shape)} (expected {shape}) or is non-finite')
    return {i: cache[i] for i in ids}


def make_loader(path, resolution, batch_size, seed, num_workers, accelerator):
    data = ImageMetadataDataset(path, resolution)
    if len(data) < batch_size:
        raise ValueError(f'{path} has fewer records than batch-size {batch_size}')
    sampler = EpochPermutationSampler(data, seed=seed)
    loader = DataLoader(data, batch_size=batch_size, sampler=sampler,
                        drop_last=True, num_workers=num_workers,
                        pin_memory=accelerator.device.type == 'cuda', collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    return loader, sampler


def save_training_state(accelerator, output, step, epoch, batches_in_epoch, seed, contract, counters):
    path = output / f'checkpoint-{step}'
    accelerator.save_state(str(path))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (path / 'trainer_cursor.json').write_text(json.dumps({
            'step': step, 'epoch': epoch, 'batches_in_epoch': batches_in_epoch,
            'seed': seed, 'world_size': accelerator.num_processes,
            'contract': contract, 'counters': counters,
            'note': 'Requires same world size / train data / settings to resume.'
        }, indent=2))
    accelerator.wait_for_everyone()


def validate(a):
    if a.max_steps <= 0 or a.batch_size <= 0 or a.grad_accum <= 0:
        raise ValueError('max-steps, batch-size and grad-accum must be positive')
    if a.anchor_batch_size <= 0 or a.num_workers < 0:
        raise ValueError('anchor-batch-size must be positive and num-workers nonnegative')
    if not all(math.isfinite(x) for x in [a.lr, a.text_lr, a.anchor_weight, a.margin_weight, a.margin]):
        raise ValueError('Training scalars must be finite')
    if a.margin < 0:
        raise ValueError('margin must be nonnegative')
    if a.lr <= 0 or a.text_lr <= 0:
        raise ValueError('Learning rates must be positive')
    if a.anchor_weight < 0 or a.margin_weight < 0:
        raise ValueError('Loss weights must be nonnegative')
    if a.method != 'standard' and not a.trusted_dir:
        raise ValueError('--trusted-dir required for defense methods')
    if a.method == 'vaccine' and not (a.vocabulary or a.negative_map):
        raise ValueError('--vocabulary (or --negative-map keys) required for vaccine')
    if a.method == 'semantic_margin' and not a.negative_map:
        raise ValueError('--negative-map required for semantic_margin')
    if a.log_every <= 0 or a.checkpoint_every < 0:
        raise ValueError('log-every must be >0 and checkpoint-every >=0')
    if a.resolution <= 0 or a.resolution % 8:
        raise ValueError('--resolution must be a positive multiple of 8')


def main():
    a = parse_args()
    validate(a)
    configure_determinism(a.deterministic)
    inputs = inspect_inputs(a.model, a.train_dir, a.trusted_dir, a.negative_map, require_gpu=not a.dry_run or (bool(os.environ.get('SLURM_JOB_ID')) and torch.cuda.is_available()),
                            output_dir=a.output_dir)
    if a.dry_run:
        print(json.dumps(inputs, indent=2))
        return
    output = Path(a.output_dir).resolve()
    prepare_output(output)
    accelerator = Accelerator(mixed_precision=a.mixed_precision,
                              gradient_accumulation_plugin=accumulation_plugin(a.grad_accum))
    contract = resume_contract(a, inputs, accelerator.num_processes)
    cursor = None
    if a.resume_from:
        cursor = json.loads((Path(a.resume_from) / 'trainer_cursor.json').read_text())
        check_resume_contract(cursor.get('contract', {}), contract)
        if int(cursor['step']) >= a.max_steps:
            raise ValueError('Resume max-steps must exceed the saved optimizer step')
    session_start = time.perf_counter()
    counters = dict(train_exposures=0, trusted_exposures=0, denoiser_forward_examples=0,
                    skipped_updates=0, wall_seconds=0.0, peak_vram_bytes=0)
    if cursor:
        counters.update(cursor['counters'])
    previous_wall = counters['wall_seconds']
    torch.cuda.reset_peak_memory_stats()
    if accelerator.device.type != 'cuda':
        raise RuntimeError('This real-image full training run requires an allocated CUDA GPU.')
    if a.mixed_precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('bf16 unavailable; use --mixed-precision fp16')
    # device_specific: each DDP rank draws its own noise/timesteps (identical draws across ranks
    # would correlate the gradient estimates). Rank RNG states are saved/restored by Accelerate.
    set_seed(a.seed, device_specific=True)
    dtype = {'no': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[a.mixed_precision]

    model_dir = a.model
    tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder='tokenizer', local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(model_dir, subfolder='text_encoder', local_files_only=True)
    vae = AutoencoderKL.from_pretrained(model_dir, subfolder='vae', local_files_only=True)
    scheduler = DDPMScheduler.from_pretrained(model_dir, subfolder='scheduler', local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder='unet', local_files_only=True)
    if a.init_unet == 'random':
        unet = UNet2DConditionModel.from_config(unet.config)

    vae.to(accelerator.device, dtype=dtype).eval().requires_grad_(False)
    encoder.requires_grad_(bool(a.train_text_encoder))
    encoder.train(a.train_text_encoder)
    unet.train().requires_grad_(True)
    if a.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if a.train_text_encoder and hasattr(encoder, 'gradient_checkpointing_enable'):
            encoder.gradient_checkpointing_enable()

    if a.optimizer == 'adamw8bit':
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ImportError('adamw8bit requires CUDA-compatible bitsandbytes; choose --optimizer adamw') from e
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    groups = [{'params': list(unet.parameters()), 'lr': a.lr}]
    if a.train_text_encoder:
        groups.append({'params': list(encoder.parameters()), 'lr': a.text_lr})
    optimizer = optimizer_cls(groups)

    train_loader, train_sampler = make_loader(a.train_dir, a.resolution, a.batch_size,
                                              a.seed, a.num_workers, accelerator)
    if a.method != 'standard':
        anchor_loader, anchor_sampler = make_loader(a.trusted_dir, a.resolution,
            a.anchor_batch_size, a.seed + 817, a.num_workers, accelerator)
        if a.train_text_encoder:
            unet, encoder, optimizer, train_loader, anchor_loader = accelerator.prepare(
                unet, encoder, optimizer, train_loader, anchor_loader)
        else:
            unet, optimizer, train_loader, anchor_loader = accelerator.prepare(
                unet, optimizer, train_loader, anchor_loader)
            encoder = encoder.to(accelerator.device, dtype=dtype)
    else:
        if a.train_text_encoder:
            unet, encoder, optimizer, train_loader = accelerator.prepare(
                unet, encoder, optimizer, train_loader)
        else:
            unet, optimizer, train_loader = accelerator.prepare(unet, optimizer, train_loader)
            encoder = encoder.to(accelerator.device, dtype=dtype)
        anchor_loader = anchor_sampler = None

    if not a.train_text_encoder:
        encoder.eval()
    require_nonempty_loaders(train_loader, anchor_loader)
    negative_map = json.loads(Path(a.negative_map).read_text()) if a.negative_map else {}
    vocabulary = a.vocabulary or sorted(negative_map)
    latent_cache = load_latent_cache(a.latent_cache, a.train_dir, a.resolution) if a.latent_cache else None
    sample_w, sample_w_mean = (load_sample_weights(a.sample_weights, a.train_dir)
                               if a.sample_weights else (None, None))
    output = Path(a.output_dir).resolve()
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        config = dict(vars(a))
        config['world_size'] = accelerator.num_processes
        config['unet_trainable_parameters'] = sum(p.numel() for p in accelerator.unwrap_model(unet).parameters() if p.requires_grad)
        config['text_encoder_trainable_parameters'] = sum(p.numel() for p in accelerator.unwrap_model(encoder).parameters() if p.requires_grad)
        config['cuda_device'] = torch.cuda.get_device_name(accelerator.device)
        config['resume_contract'] = contract
        config['effective_batch_size'] = a.batch_size * a.grad_accum * accelerator.num_processes
        config['ema_enabled'] = False
        if a.latent_cache:
            config['latent_cache_sha256'] = contract['latent_cache_sha256']
        if a.method == 'vaccine':
            config['vaccine_vocabulary'] = vocabulary
        if a.sample_weights:
            config['sample_weights_sha256'] = sha256(Path(a.sample_weights).read_bytes()).hexdigest()
            config['sample_weights_dataset_mean'] = sample_w_mean
        config['slurm_job_id'] = os.environ.get('SLURM_JOB_ID')
        (output / 'preflight.json').write_text(json.dumps(inputs, indent=2))
        config['train_metadata_sha256'] = sha256((Path(a.train_dir) / 'metadata.jsonl').read_bytes()).hexdigest()
        if a.trusted_dir:
            config['trusted_metadata_sha256'] = sha256((Path(a.trusted_dir) / 'metadata.jsonl').read_bytes()).hexdigest()
        for pkg in ['torch', 'accelerate', 'diffusers', 'transformers', 'safetensors']:
            try: config[f'version_{pkg}'] = version(pkg)
            except PackageNotFoundError: config[f'version_{pkg}'] = 'not-installed'
        (output / 'config.json').write_text(json.dumps(config, indent=2))
        (output / 'run_manifest.json').write_text(json.dumps(config, indent=2))
    accelerator.wait_for_everyone()

    step, epoch, batches_in_epoch = 0, 0, 0
    if a.resume_from:
        checkpoint = Path(a.resume_from).resolve()
        # Cursor and immutable data/settings contract were checked before loading models.
        if cursor['seed'] != a.seed or cursor['world_size'] != accelerator.num_processes:
            raise ValueError('Resume seed and world_size must match checkpoint')
        accelerator.load_state(str(checkpoint))
        step, epoch, batches_in_epoch = (int(cursor[k]) for k in
                                         ('step', 'epoch', 'batches_in_epoch'))
        epoch, batches_in_epoch = normalize_cursor(epoch, batches_in_epoch, len(train_loader))

    while step < a.max_steps:
        set_loader_epoch(train_loader, train_sampler, epoch)
        if anchor_sampler is not None:
            anchor_wrap = batches_in_epoch // len(anchor_loader)
            anchor_offset = batches_in_epoch % len(anchor_loader)
            set_loader_epoch(anchor_loader, anchor_sampler, epoch * 1_000_000 + anchor_wrap)
            anchor_iter = iter(accelerator.skip_first_batches(anchor_loader, anchor_offset)
                               if anchor_offset else anchor_loader)
        else:
            anchor_iter = None
        current_loader = train_loader
        if batches_in_epoch:
            if batches_in_epoch > len(train_loader):
                raise ValueError('Checkpoint training cursor exceeds DataLoader length')
            current_loader = accelerator.skip_first_batches(train_loader, batches_in_epoch)
        for batch in current_loader:
            with accelerator.accumulate(unet, *([encoder] if a.train_text_encoder else [])):
                pixels = batch['pixel_values'].to(accelerator.device)
                batch_w = (torch.tensor([sample_w[i] for i in batch['ids']]) if sample_w is not None else None)
                batch_lat = (torch.stack([latent_cache[i] for i in batch['ids']]) if latent_cache is not None else None)
                loss = training_loss(unet, scheduler, vae, encoder, tokenizer,
                    pixels, batch['texts'], dtype, a.train_text_encoder, batch_w, batch_lat)
                raw_loss = loss.detach().float().item()
                if anchor_iter is not None:
                    try:
                        anchors = next(anchor_iter)
                    except StopIteration:
                        anchor_wrap += 1
                        set_loader_epoch(anchor_loader, anchor_sampler, epoch * 1_000_000 + anchor_wrap)
                        anchor_iter = iter(anchor_loader)
                        anchors = next(anchor_iter)
                    a_pixels = anchors['pixel_values'].to(accelerator.device)
                    if a.method == 'vaccine':
                        anchor_loss, margin_loss, n_rival = vaccine_margin_losses(
                            unet, scheduler, vae, encoder, tokenizer, a_pixels, anchors['texts'],
                            anchors['concepts'], vocabulary, dtype, a.train_text_encoder, a.margin)
                        loss = loss + a.anchor_weight * anchor_loss + a.margin_weight * margin_loss
                    elif a.method == 'semantic_margin':
                        negatives = anchor_negative_texts(anchors, negative_map)
                        anchor_loss, margin_loss = trusted_margin_losses(
                            unet, scheduler, vae, encoder, tokenizer, a_pixels,
                            anchors['texts'], negatives, dtype, a.train_text_encoder, a.margin)
                        loss = loss + a.anchor_weight * anchor_loss + a.margin_weight * margin_loss
                    else:
                        loss = loss + a.anchor_weight * training_loss(
                            unet, scheduler, vae, encoder, tokenizer, a_pixels,
                            anchors['texts'], dtype, a.train_text_encoder)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss; aborting without a success marker')
                accelerator.backward(loss)
                counters['train_exposures'] += len(batch['texts']) * accelerator.num_processes
                anchor_n = len(anchors['texts']) if anchor_iter is not None else 0
                counters['trusted_exposures'] += anchor_n * accelerator.num_processes
                per_anchor = {'semantic_margin': 2, 'vaccine': 1 + len(vocabulary) - 1}.get(a.method, 1)
                counters['denoiser_forward_examples'] += (len(batch['texts']) + anchor_n * per_anchor) * accelerator.num_processes
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        list(unet.parameters()) + (list(encoder.parameters()) if a.train_text_encoder else []), 1.0)
                    if not torch.isfinite(grad_norm) and a.mixed_precision != 'fp16':
                        raise FloatingPointError('Nonfinite gradient norm')
                    if grad_norm == 0:
                        raise FloatingPointError('Zero gradient norm')
                    probe = accelerator.unwrap_model(unet).conv_out.weight
                    before_update = probe.detach().clone()
                optimizer.step()
                if accelerator.sync_gradients:
                    skipped = accelerator.optimizer_step_was_skipped
                    if skipped:
                        counters['skipped_updates'] += 1
                    else:
                        update_max = float((probe.detach() - before_update).abs().max())
                        if not math.isfinite(update_max) or update_max == 0:
                            raise FloatingPointError('UNet output weights did not change by a finite amount')
                optimizer.zero_grad(set_to_none=True)
            batches_in_epoch += 1
            if optimizer_update_completed(accelerator.sync_gradients, accelerator.optimizer_step_was_skipped):
                step += 1
                counters['wall_seconds'] = previous_wall + time.perf_counter() - session_start
                counters['peak_vram_bytes'] = max(counters['peak_vram_bytes'], torch.cuda.max_memory_allocated())
                event = dict(optimizer_step=step, epoch=epoch, batches_in_epoch=batches_in_epoch,
                             mixed_loss=float(loss.detach().float()), untrusted_loss=raw_loss,
                             gradient_norm=float(grad_norm), weight_update_max=update_max, **counters)
                if accelerator.is_main_process:
                    with (output / 'training_log.jsonl').open('a') as log:
                        log.write(json.dumps(event) + '\n')
                if accelerator.is_main_process and (step == 1 or step % a.log_every == 0):
                    print(json.dumps({'optimizer_step': step, 'epoch': epoch,
                                      'mixed_loss': round(float(loss.detach().float()), 6),
                                      'untrusted_loss': round(raw_loss, 6)}), flush=True)
                if a.checkpoint_every and step % a.checkpoint_every == 0:
                    save_training_state(accelerator, output, step, epoch, batches_in_epoch, a.seed, contract, counters)
            if step >= a.max_steps:
                break
        if batches_in_epoch >= len(train_loader):
            epoch += 1
            batches_in_epoch = 0

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(unet).save_pretrained(output / 'unet', safe_serialization=True)
        if a.train_text_encoder:
            accelerator.unwrap_model(encoder).save_pretrained(output / 'text_encoder', safe_serialization=True)
        counters['wall_seconds'] = previous_wall + time.perf_counter() - session_start
        (output / 'metrics.json').write_text(json.dumps(dict(optimizer_steps=step,
            gpu_hours=counters['wall_seconds'] * accelerator.num_processes / 3600, **counters), indent=2))
        (output / 'completed.json').write_text(json.dumps({'optimizer_steps': step,
             'epoch': epoch, 'batches_in_epoch': batches_in_epoch, 'model': a.model,
             'method': a.method, 'trained_unet_all_parameters': True,
             'text_encoder_trained': bool(a.train_text_encoder),
             'vae_trained': False, 'seed': a.seed}, indent=2))
    accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
