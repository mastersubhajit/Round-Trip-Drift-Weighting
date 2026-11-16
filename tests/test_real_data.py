import csv,json,sys
from pathlib import Path
import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
from data_utils import ImageMetadataDataset, concept_negative_caption, anchor_negative_texts
from attach_nightshade import attach
from paired_dataset import load_and_validate,export


def write_im(path,col=(200,100,40)):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',(24,20),col).save(path)


def test_caption_minimal_edit():
    assert concept_negative_caption('A dog and a ball','dog','cat')=='A cat and a ball'
    with pytest.raises(ValueError):concept_negative_caption('A dog and a ball','cat','dog')


def test_actual_image_pixels_and_anchor_text(tmp_path):
    d=tmp_path/'data';write_im(d/'x.png')
    (d/'metadata.jsonl').write_text(json.dumps({'file_name':'x.png','text':'A dog in a park','concept':'dog'})+'\n')
    ds=ImageMetadataDataset(d,size=32)
    ex=ds[0]
    assert ex['pixel_values'].shape==(3,32,32)
    assert ex['pixel_values'].max()<=1 and ex['pixel_values'].min()>=-1
    assert anchor_negative_texts({'texts':[ex['text']],'concepts':[ex['concept']],'negative_texts':['']},{'dog':'cat'})==['A cat in a park']


def test_no_poison_outside_training(tmp_path):
    write_im(tmp_path/'a.png');write_im(tmp_path/'b.png')
    r={'id':'x','clean_path':'a.png','caption':'a dog','concept':'dog','split':'trusted',
       'source_id':'source1','permission':True,'license':'verified'}
    (tmp_path/'manifest.jsonl').write_text(json.dumps(r)+'\n')
    with (tmp_path/'pairs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','poison_path','attacker_target']);w.writeheader()
        w.writerow({'id':'x','poison_path':'b.png','attacker_target':'cat'})
    with pytest.raises(ValueError,match='Poison cannot enter'):
        attach(tmp_path/'manifest.jsonl',tmp_path/'pairs.csv',tmp_path/'out.jsonl')
