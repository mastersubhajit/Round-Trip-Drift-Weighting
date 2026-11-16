#!/usr/bin/env python3
"""Quantify real paired-image perturbations; does not certify genuine Nightshade.

Reports actual pixel magnitude and SHA256 for authenticated experiment provenance.
For an attack-effective test, the poisoned *undefended model* must fail later.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import numpy as np
from PIL import Image
from paired_dataset import load_and_validate


def pair_stats(rows):
    reports=[]
    for r in rows:
        if r['_poison'] is None:continue
        with Image.open(r['_clean']) as ca,Image.open(r['_poison']) as pa:
            if ca.size!=pa.size:raise ValueError(f'Dimensions differ for {r["id"]}')
            c=np.asarray(ca.convert('RGB'),dtype=np.float32)
            p=np.asarray(pa.convert('RGB'),dtype=np.float32)
        d=np.abs(c-p)
        mse=float(np.mean((c-p)**2))
        report={'id':r['id'],'concept':r['concept'],'attack':r['attack'],
                'attacker_target':r['attacker_target'],
                'max_channel_Linf_255':float(d.max()),'mean_absolute_pixel_255':float(d.mean()),
                'mse_pixel_255':mse,'psnr_db':None if mse==0 else float(20*np.log10(255)-10*np.log10(mse)),
                'clean_sha256':sha256(Path(r['_clean']).read_bytes()).hexdigest(),
                'poison_sha256':sha256(Path(r['_poison']).read_bytes()).hexdigest()}
        reports.append(report)
    if not reports:raise ValueError('No poisoned image pairs in manifest')
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows,summary=load_and_validate(a.manifest)
    report=pair_stats(rows)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps({'n':len(report),'pairs':report},indent=2))
    print(f'Audited {len(report)} REAL image pairs; perturbation size is not evidence of attack success')
if __name__=='__main__':main()
