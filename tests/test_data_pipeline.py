import csv,json,sys
from pathlib import Path
import pytest
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
from data_utils import EpochPermutationSampler
from paired_dataset import load_and_validate,export


def make_manifest(tmp_path):
    rows=[]
    for i,(split,concept) in enumerate([('train','dog'),('train','cat'),('trusted','dog'),('test','dog')]):
        pic=tmp_path/f'{i}.png';Image.new('RGB',(32,32),(i*40,10,200)).save(pic)
        row=dict(id=f'id{i}',clean_path=pic.name,caption=f'a {concept}',concept=concept,
                 split=split,source_id=f'source{i}',license='owner authorized',permission=True)
        if i==0:
            poison=tmp_path/'poison.png';Image.new('RGB',(32,32),(5,220,7)).save(poison)
            row.update(poison_path=poison.name,attack='nightshade_research',attacker_target='cat')
        rows.append(row)
    p=tmp_path/'manifest.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return p


def test_epoch_sampler_deterministic_and_distinct():
    one=EpochPermutationSampler(list(range(40)),seed=2718)
    perm0=list(iter(one));one.set_epoch(1);perm1=list(iter(one));one.set_epoch(0)
    assert perm0==list(iter(one)) and sorted(perm0)==list(range(40)) and perm0!=perm1


def test_denoising_margin_gradient_direction():
    from training_objectives import contrastive_denoising_margin,per_example_denoising_loss
    pos=torch.tensor([1.0],requires_grad=True)
    neg=torch.tensor([2.0],requires_grad=True)
    loss=contrastive_denoising_margin(pos,neg)
    loss.backward()
    assert pos.grad.item()>0  # decrease positive error
    assert neg.grad.item()<0  # increase negative error
    pred=torch.zeros((2,3,4,4),requires_grad=True)
    target=torch.ones_like(pred)
    assert torch.allclose(per_example_denoising_loss(pred,target), torch.ones(2))


def test_audit_attack_different_real_pixels(tmp_path):
    from audit_attack import pair_stats
    rows,_=load_and_validate(make_manifest(tmp_path))
    stats=pair_stats(rows)
    assert len(stats)==1 and stats[0]['max_channel_Linf_255']>0
    assert stats[0]['clean_sha256']!=stats[0]['poison_sha256']
