"""Dataset utilities. Imported by CPU-only unit tests and GPU training."""
from __future__ import annotations
import json, re
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset


def concept_negative_caption(caption: str, concept: str, negative: str) -> str:
    """Minimal contrast that changes ONLY the stated concept; fail closed otherwise."""
    if not (caption and concept and negative):
        raise ValueError("caption, concept and negative are mandatory")
    pattern = re.compile(r"(?<![\w])" + re.escape(concept) + r"(?![\w])", flags=re.IGNORECASE)
    new, n = pattern.subn(negative, caption, count=1)
    if n == 0:
        raise ValueError(f"Concept {concept!r} absent from caption {caption!r}; annotate negative_text manually")
    return new


def load_jsonl(path):
    rows=[]
    with open(path, encoding='utf8') as f:
        for i, line in enumerate(f,1):
            if line.strip():
                r=json.loads(line)
                if not isinstance(r,dict):raise ValueError(f"record {i} is not an object")
                rows.append(r)
    if not rows:raise ValueError(f"No records in {path}")
    return rows


class ImageMetadataDataset(Dataset):
    """Uses real RGB image files in ImageFolder metadata format; no synthetic data."""
    def __init__(self, path, size=512):
        self.root=Path(path)
        self.rows=load_jsonl(self.root/'metadata.jsonl')
        self.size=int(size)
        if self.size<=0 or self.size%8!=0:raise ValueError('size must be a positive multiple of 8')
        for r in self.rows:
            if not {'file_name','text'} <= set(r):raise ValueError('file_name and text required')
            p=(self.root/r['file_name']).resolve()
            if not p.is_file() or not p.is_relative_to(self.root.resolve()):
                raise ValueError(f'Invalid image path: {p}')

    def __len__(self):return len(self.rows)

    def __getitem__(self, i):
        r=self.rows[i]
        with Image.open(self.root/r['file_name']) as img:
            img=ImageOps.exif_transpose(img).convert('RGB')
            # Fit image with same deterministic geometry for clean / poison counterparts.
            w,h=img.size
            scale=max(self.size/w,self.size/h)
            w2,h2=int(round(w*scale)),int(round(h*scale))
            img=img.resize((w2,h2), Image.Resampling.BICUBIC)
            left=(w2-self.size)//2; top=(h2-self.size)//2
            img=img.crop((left,top,left+self.size,top+self.size))
            px=torch.from_numpy(np.asarray(img,dtype=np.float32).copy()).permute(2,0,1)/127.5-1
        return {'pixel_values':px,'text':r['text'],'concept':r.get('concept',''),
                'negative_text':r.get('negative_text',''),'id':r.get('id','')}


def collate(batch):
    return {'pixel_values':torch.stack([b['pixel_values'] for b in batch]),
            'texts':[b['text'] for b in batch], 'concepts':[b['concept'] for b in batch],
            'negative_texts':[b['negative_text'] for b in batch], 'ids':[b['id'] for b in batch]}


def anchor_negative_texts(batch, mapping):
    out=[]
    for text, concept, explicit in zip(batch['texts'],batch['concepts'],batch['negative_texts']):
        if explicit:out.append(explicit)
        else:
            if concept not in mapping:raise ValueError(f'No negative concept mapping for {concept!r}')
            out.append(concept_negative_caption(text,concept,mapping[concept]))
    return out


class EpochPermutationSampler(torch.utils.data.Sampler):
    """Per-epoch seeded permutation for reproducible shuffles across restarts.

    Trainer owns the epoch; data have no stochastic transforms. For strict exact
    bit-for-bit reproducibility, also keep GPU count/framework/CUDA versions fixed.
    """
    def __init__(self, data, seed=42):
        self.data = data
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data), generator=g).tolist())

    def __len__(self):
        return len(self.data)
