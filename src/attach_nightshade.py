#!/usr/bin/env python3
"""Attach generated research Nightshade image paths using explicit verified ID pairs.

Mapping CSV must have id,poison_path,attacker_target. This never generates poisons,
never changes source captions, and never permits altered validation/test/trusted records.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from PIL import Image
from data_utils import load_jsonl


def attach(manifest,map_csv,out):
    manifest=Path(manifest).resolve(); map_csv=Path(map_csv).resolve()
    records=load_jsonl(manifest); known={r['id']:r for r in records}
    pairs={}
    with open(map_csv,newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):
            rid=r['id']; p=(map_csv.parent / r['poison_path']).resolve()
            if rid in pairs or rid not in known:raise ValueError(f'Duplicate or unknown ID {rid}')
            if known[rid]['split']!='train':raise ValueError(f'Poison cannot enter {known[rid]["split"]} split')
            if not p.is_file():raise FileNotFoundError(p)
            if not r['attacker_target']:raise ValueError('attacker_target required')
            with Image.open(p) as a,Image.open((manifest.parent / known[rid]['clean_path']).resolve()) as b:
                if a.size!=b.size:raise ValueError(f'Paired dimensions differ: {rid}')
            pairs[rid]=(str(p),r['attacker_target'])
    if not pairs:raise ValueError('Mapping CSV empty')
    for r in records:
        # Resolve against the input manifest before writing to a different directory.
        r['clean_path'] = str((manifest.parent / r['clean_path']).resolve())
        if r.get('poison_path'):
            r['poison_path'] = str((manifest.parent / r['poison_path']).resolve())
        if r['id'] in pairs:
            r['poison_path'],r['attacker_target']=pairs[r['id']]
            r['attack']='nightshade-research'
    Path(out).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf8')
    return len(pairs)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--pairs-csv',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args();print('Attached',attach(a.manifest,a.pairs_csv,a.out),'poison pairs')
