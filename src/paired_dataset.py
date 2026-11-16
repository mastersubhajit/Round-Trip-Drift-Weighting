#!/usr/bin/env python3
"""Manifest validator/exporter for *real* clean/poisoned image-caption pairs.

Does NOT generate Nightshade poisons or claim to implement a defense.
Input: JSONL with id,clean_path,poison_path,caption,concept,split,source_id,
license,permission,attack,attacker_target. Paths relative to input manifest.
Only paired records with actual poison files can be selected for contaminated training.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import shutil
import sys
from PIL import Image

SPLITS={'train','validation','test','trusted'}
REQUIRED={'id','clean_path','caption','concept','split','source_id','license','permission'}

def load_and_validate(path:Path):
    rows=[]; ids=set(); hashes={}; counts=Counter(); warnings=[]
    path=path.resolve()
    with path.open(encoding='utf8') as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            try: r=json.loads(line)
            except Exception as e: raise ValueError(f'line {n}: invalid JSON: {e}') from e
            missing=REQUIRED-set(r)
            if missing: raise ValueError(f'line {n}: missing {sorted(missing)}')
            for key in ('id','caption','concept','source_id','license','clean_path'):
                if not isinstance(r[key],str) or not r[key].strip(): raise ValueError(f'line {n}: invalid {key}')
            if r['permission'] is not True: raise ValueError(f'line {n}: permission must be JSON boolean true (verified by dataset owner)')
            if r['split'] not in SPLITS: raise ValueError(f'line {n}: invalid split {r["split"]!r}')
            if r['id'] in ids: raise ValueError(f'line {n}: duplicate record id {r["id"]}')
            ids.add(r['id'])
            r['_clean']=str((path.parent / r['clean_path']).resolve())
            cp=Path(r['_clean'])
            if not cp.is_file(): raise FileNotFoundError(f'line {n}: missing clean image {cp}')
            with Image.open(cp) as im:
                im.verify()
            digest=sha256(cp.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest]!=r['split']:
                raise ValueError(f'line {n}: same image bytes appear in different splits')
            hashes[digest]=r['split']
            pp=r.get('poison_path')
            if pp:
                if r['split'] != 'train':
                    raise ValueError(f'line {n}: Poison cannot enter trusted/validation/test splits')
                poison_file=(path.parent / pp).resolve()
                if not poison_file.is_file(): raise FileNotFoundError(f'line {n}: missing poisoned image {poison_file}')
                with Image.open(poison_file) as im:
                    im.verify()
                r['_poison']=str(poison_file)
                if sha256(poison_file.read_bytes()).hexdigest()==digest:
                    warnings.append(f'line {n}: paired clean and poison file bytes are identical')
                if not r.get('attack') or not r.get('attacker_target'):
                    raise ValueError(f'line {n}: attack and attacker_target mandatory when poison_path supplied')
            else:
                r['_poison']=None
            counts[(r['split'],r['concept'])]+=1
            rows.append(r)
    if not rows: raise ValueError('Manifest is empty')
    source_splits=defaultdict(set)
    for r in rows:source_splits[r['source_id']].add(r['split'])
    overlapping_sources={k:sorted(v) for k,v in source_splits.items() if len(v)>1}
    if overlapping_sources:
        warnings.append(f'{len(overlapping_sources)} source IDs appear in multiple splits; assess independence')
    summary={'n':len(rows),'overlapping_source_ids_across_splits':overlapping_sources, 'by_split_concept':{f'{s}:{c}':v for (s,c),v in sorted(counts.items())},
             'paired_poison_records':sum(r['_poison'] is not None for r in rows), 'warnings':warnings}
    return rows,summary

def select_poison_ids(eligible_ids, k, seed):
    """THE poison-selection rule (used by export and by attack variants that must poison the same records).
    Seeded shuffle of the id-sorted eligible records, then prefix: poison sets are NESTED across doses
    (k=25 ⊂ 50 ⊂ 100 ...) for a fixed seed, so a dose-response curve compares supersets, not unrelated draws."""
    order=sorted(eligible_ids); random.Random(seed).shuffle(order)
    return order[:k]


def selection_from_manifest(manifest, concept, k, seed):
    """Ids that `export(arm='poisoned', target_concept=concept, poison_count=k, seed=seed)` selects from MANIFEST.
    Attack variants (other eps, adaptive, dirty-label) are generated for exactly these records, so every attack is
    compared on the same images and the same clean remainder."""
    eligible=[]
    for line in Path(manifest).read_text(encoding='utf8').splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        if r.get('split')=='train' and r.get('concept')==concept and r.get('poison_path'):
            eligible.append(r['id'])
    if k>len(eligible): raise ValueError(f'{manifest}: only {len(eligible)} poisoned {concept} records, need {k}')
    return select_poison_ids(eligible,k,seed)


def export(rows, output:Path, arm:str, target_concept:str, fraction:float, seed:int, poison_count=None, heldout_split='test'):
    if arm not in {'clean','poisoned','trusted_only','heldout_clean'}: raise ValueError('unknown arm')
    if not 0<=fraction<=1: raise ValueError('poison fraction must be between 0 and 1')
    if arm=='poisoned' and not target_concept: raise ValueError('target_concept required')
    selected_split = ('trusted' if arm=='trusted_only' else heldout_split if arm=='heldout_clean' else 'train')
    if arm=='heldout_clean' and heldout_split not in {'validation','test'}:
        raise ValueError('heldout_split must be test or validation')
    relevant=[r for r in rows if r['split']==selected_split]
    if not relevant: raise ValueError(f'no data in split for {arm}')
    target=[r for r in relevant if r['concept']==target_concept]
    n_poison=(poison_count if poison_count is not None else round(fraction*len(target))) if arm=='poisoned' else 0
    if n_poison<0 or n_poison>len(target):raise ValueError('poison count must lie between 0 and number of target concept records')
    eligible=[r for r in target if r['_poison']]
    if n_poison>len(eligible):
        raise ValueError(f'Need {n_poison} poisoned counterparts for {target_concept}; only {len(eligible)} available')
    selected=set(select_poison_ids([r['id'] for r in eligible],n_poison,seed))
    out=output/'train'; out.mkdir(parents=True,exist_ok=False)
    with (out/'metadata.jsonl').open('w',encoding='utf8') as f:
        for r in sorted(relevant,key=lambda r:r['id']):
            poisoned=r['id'] in selected
            src=Path(r['_poison'] if poisoned else r['_clean'])
            # Preserve extension. A stable, collision-resistant filename.
            filename=f"{sha256(r['id'].encode('utf8')).hexdigest()[:20]}{src.suffix.lower()}"
            shutil.copyfile(src,out/filename)
            # Keep only file_name and text for maximum imagefolder training compatibility.
            f.write(json.dumps({'file_name':filename,'text':r['caption'],'id':r['id'],'concept':r['concept'],'source_id':r['source_id'],'is_poisoned':poisoned,'image_sha256':sha256(src.read_bytes()).hexdigest(),
                                'negative_text':r.get('negative_text','')},ensure_ascii=False)+'\n')
    info={'arm':arm,'seed':seed,'split':selected_split,'train_records':len(relevant),
          'target_concept':target_concept or None, 'target_concept_records':len(target),
          'targeted_poison_records':n_poison,
          'targeted_poison_fraction':n_poison/len(target) if target else 0,
          'global_poison_fraction':n_poison/len(relevant),
          'selected_poison_ids':sorted(selected),
          'record_provenance':[{'id':r['id'],'concept':r['concept'],'source_id':r['source_id'],
                                'is_poisoned':r['id'] in selected,'split':r['split'],
                                'attack':r.get('attack') if r['id'] in selected else None,
                                'permission':r['permission'],'license':r['license'],
                                'rights_evidence':r.get('rights_evidence'),
                                'source_url':r.get('source_url'),
                                'attack_metadata':r.get('attack_metadata') if r['id'] in selected else None}
                               for r in relevant]}
    (output/'audit.json').write_text(json.dumps(info,indent=2,ensure_ascii=False))
    return info

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--arm',choices=['clean','poisoned','trusted_only','heldout_clean'],default='clean')
    p.add_argument('--heldout-split',choices=['validation','test'],default='test')
    p.add_argument('--target-concept')
    p.add_argument('--poison-fraction',type=float,default=0.0)
    p.add_argument('--poison-count',type=int,default=None,help='Exact number of poisoned pairs for target concept (overrides fraction)')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--audit-only',action='store_true')
    a=p.parse_args(); rows,summary=load_and_validate(a.manifest)
    print(json.dumps(summary,indent=2))
    if not a.audit_only:
        if a.out is None: p.error('--out required unless --audit-only')
        if a.out.exists(): p.error(f'output already exists: {a.out}; choose a fresh directory')
        details=export(rows,a.out,a.arm,a.target_concept or '',a.poison_fraction,a.seed,a.poison_count,a.heldout_split)
        print(json.dumps({k:v for k,v in details.items() if k!='record_provenance'},indent=2))

if __name__=='__main__':main()
