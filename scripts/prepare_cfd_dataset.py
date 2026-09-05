from __future__ import annotations
import argparse, hashlib, json, os, random, re, shutil, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from crackmeanflow.common import discover_required_splits, build_dataset_identity, audit_group_integrity

IMG_EXTS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
IMG_RE=re.compile(r'^(?P<parent>[^_]+)_img_(?P<crop>.+)$')
MSK_RE=re.compile(r'^(?P<parent>[^_]+)_msk_(?P<crop>.+)$')

def sha256_file(path: Path, chunk=1024*1024):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(chunk), b''): h.update(b)
    return h.hexdigest()

def _normalized(stem: str, rx: re.Pattern):
    m=rx.match(stem)
    if not m: raise RuntimeError(f'filename does not match CFD naming contract: {stem!r}')
    return f"{m.group('parent')}_{m.group('crop')}", m.group('parent')

def _discover_dir(root: Path):
    image_dir=root/'image'; label_dir=root/'label'
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f'expected flat CFD root with image/ and label/: {root}')
    images={}; masks={}; parents={}
    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS: continue
        key,parent=_normalized(p.stem,IMG_RE)
        if key in images: raise RuntimeError(f'duplicate normalized image key={key}')
        images[key]=p; parents[key]=parent
    for p in sorted(label_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS: continue
        key,parent=_normalized(p.stem,MSK_RE)
        if key in masks: raise RuntimeError(f'duplicate normalized mask key={key}')
        masks[key]=p
        if key in parents and parents[key]!=parent: raise RuntimeError(f'parent mismatch for {key}')
    return images,masks,parents

def _discover_zip(path: Path):
    zf=zipfile.ZipFile(path)
    image_entries=[n for n in zf.namelist() if 'image' in Path(n).parts[:-1] and not n.endswith('/') and Path(n).suffix.lower() in IMG_EXTS]
    mask_entries=[n for n in zf.namelist() if 'label' in Path(n).parts[:-1] and not n.endswith('/') and Path(n).suffix.lower() in IMG_EXTS]
    images={}; masks={}; parents={}
    for n in sorted(image_entries):
        p=Path(n); key,parent=_normalized(p.stem,IMG_RE)
        if key in images: raise RuntimeError(f'duplicate normalized image key={key}')
        images[key]=n; parents[key]=parent
    for n in sorted(mask_entries):
        p=Path(n); key,parent=_normalized(p.stem,MSK_RE)
        if key in masks: raise RuntimeError(f'duplicate normalized mask key={key}')
        masks[key]=n
        if key in parents and parents[key]!=parent: raise RuntimeError(f'parent mismatch for {key}')
    return zf,images,masks,parents

def _largest_remainder_counts(n, fracs):
    raw=[n*f for f in fracs]; base=[int(x) for x in raw]; rem=n-sum(base)
    order=sorted(range(len(fracs)), key=lambda i:(raw[i]-base[i], -i), reverse=True)
    for i in order[:rem]: base[i]+=1
    return base

def _assign_parents(parents, seed, train_frac, val_frac):
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac+val_frac < 1):
        raise ValueError('fractions must satisfy 0<train,val and train+val<1')
    ids=sorted(set(parents.values())); rng=random.Random(seed); rng.shuffle(ids)
    test_frac=round(1-train_frac-val_frac,12); nt,nv,nte=_largest_remainder_counts(len(ids), [train_frac,val_frac,test_frac])
    if min(nt,nv,nte)<1: raise RuntimeError(f'not enough parent groups for 3-way split: n={len(ids)} counts={(nt,nv,nte)}')
    assign={'train':set(ids[:nt]),'val':set(ids[nt:nt+nv]),'test':set(ids[nt+nv:])}
    counts=Counter(parents.values()); total=sum(counts.values()); targets={'train':total*train_frac,'val':total*val_frac,'test':total*test_frac}
    def objective(a):
        return sum(((sum(counts[x] for x in a[s])-targets[s])/max(targets[s],1.0))**2 for s in a)
    # Deterministic pairwise swaps retain the requested parent counts while making
    # crop/sample fractions much closer to 70/15/15.
    for _ in range(100):
        base=objective(assign); best=base; choice=None; split_names=('train','val','test')
        for i,s1 in enumerate(split_names):
            for s2 in split_names[i+1:]:
                for a1 in sorted(assign[s1]):
                    for a2 in sorted(assign[s2]):
                        cand={k:set(v) for k,v in assign.items()};cand[s1].remove(a1);cand[s1].add(a2);cand[s2].remove(a2);cand[s2].add(a1);score=objective(cand)
                        if score < best-1e-15: best=score;choice=cand
        if choice is None: break
        assign=choice
    return assign

def _validate_pairs(images,masks,parents):
    mi=sorted(set(masks)-set(images)); mm=sorted(set(images)-set(masks))
    if mi or mm: raise RuntimeError(f'unpaired normalized CFD keys: missing_images={mi[:10]} ({len(mi)}), missing_masks={mm[:10]} ({len(mm)})')
    if not images: raise RuntimeError('no CFD pairs discovered')
    return sorted(images)

def _copy_dir(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists(): dst.unlink()
    if mode=='copy': shutil.copy2(src,dst)
    elif mode=='hardlink': os.link(src,dst)
    elif mode=='symlink': os.symlink(src.resolve(),dst)
    else: raise ValueError(mode)

def _audit_images(output: Path):
    # Content duplicate images across splits would invalidate parent-disjoint evaluation.
    by_hash=defaultdict(list); mask_by_hash=defaultdict(list); dims=Counter(); mask_values=Counter(); fg=[]; dhashes=[]
    for split in ('train','val','test'):
        mask_map={p.stem:p for p in (output/split/'cracked'/'masks').iterdir() if p.suffix.lower() in IMG_EXTS}
        for ip in sorted((output/split/'cracked'/'images').iterdir()):
            mp=mask_map.get(ip.stem)
            if mp is None: raise RuntimeError(f'audit could not find mask for {split}/{ip.stem}')
            by_hash[sha256_file(ip)].append((split,ip.stem)); mask_by_hash[sha256_file(mp)].append((split,mp.stem))
            with Image.open(ip) as im:
                dims[tuple(im.size)]+=1
                g=im.convert('L').resize((9,8),Image.Resampling.BILINEAR);a=np.asarray(g);bits=(a[:,1:]>a[:,:-1]).reshape(-1);h64=0
                for bit in bits:h64=(h64<<1)|int(bit)
                dhashes.append((split,ip.stem,h64))
            with Image.open(mp) as m:
                a=np.asarray(m.convert('L')); vals=np.unique(a); mask_values.update(int(v) for v in vals); fg.append(float(((a>0) if int(a.max())<=1 else (a>127)).mean()))
    duplicates=[{'sha256':h,'rows':rows} for h,rows in by_hash.items() if len(rows)>1]
    cross=[x for x in duplicates if len({s for s,_ in x['rows']})>1]
    if cross: raise RuntimeError(f'exact duplicate image content crosses splits: {cross[:3]}')
    if duplicates: raise RuntimeError(f'exact duplicate image content exists in CFD: {len(duplicates)} groups; examples={duplicates[:3]}')
    near=[]
    for i,(s1,n1,h1) in enumerate(dhashes):
        for s2,n2,h2 in dhashes[i+1:]:
            if s1==s2: continue
            d=(h1^h2).bit_count()
            if d<=2: near.append({'hamming':d,'a':[s1,n1],'b':[s2,n2]})
    return {
        'image_dimensions':{f'{w}x{h}':n for (w,h),n in sorted(dims.items())},
        'mask_unique_values':sorted(mask_values),
        'foreground_ratio':{'min':min(fg),'mean':sum(fg)/len(fg),'max':max(fg)},
        'exact_duplicate_image_groups':sum(1 for v in by_hash.values() if len(v)>1),
        'exact_duplicate_mask_groups':sum(1 for v in mask_by_hash.values() if len(v)>1),
        'perceptual_cross_split_dhash_hamming_le_2_count':len(near),
        'perceptual_cross_split_examples':near[:20],
    }

def _group_fn(stem): return stem.split('_',1)[0]

def main():
    ap=argparse.ArgumentParser(description='Normalize flat CFD image/*_img_* + label/*_msk_* into parent-disjoint canonical splits.')
    ap.add_argument('--input',required=True,help='flat CFD directory or ZIP archive')
    ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--train-frac',type=float,default=.70)
    ap.add_argument('--val-frac',type=float,default=.15)
    ap.add_argument('--mode',choices=['copy','hardlink','symlink'],default='copy',help='directory input only; ZIP always extracts/copies')
    ap.add_argument('--overwrite',action='store_true')
    a=ap.parse_args(); src=Path(a.input).resolve(); out=Path(a.out).resolve()
    if out.exists():
        if not a.overwrite: raise FileExistsError(f'output exists: {out}; pass --overwrite to replace')
        shutil.rmtree(out)
    out.mkdir(parents=True)
    is_zip=src.is_file() and zipfile.is_zipfile(src)
    zf=None
    if is_zip: zf,images,masks,parents=_discover_zip(src)
    else: images,masks,parents=_discover_dir(src)
    keys=_validate_pairs(images,masks,parents); split_parents=_assign_parents(parents,a.seed,a.train_frac,a.val_frac)
    key_split={k:next(s for s,ps in split_parents.items() if parents[k] in ps) for k in keys}
    for key in keys:
        split=key_split[key]; img_dst=out/split/'cracked'/'images'/(key+Path(images[key]).suffix.lower()); msk_dst=out/split/'cracked'/'masks'/(key+Path(masks[key]).suffix.lower())
        img_dst.parent.mkdir(parents=True,exist_ok=True); msk_dst.parent.mkdir(parents=True,exist_ok=True)
        if is_zip:
            with zf.open(images[key]) as fi, img_dst.open('wb') as fo: shutil.copyfileobj(fi,fo)
            with zf.open(masks[key]) as fi, msk_dst.open('wb') as fo: shutil.copyfileobj(fi,fo)
        else:
            _copy_dir(images[key],img_dst,a.mode); _copy_dir(masks[key],msk_dst,a.mode)
    if zf is not None: zf.close()
    splits=discover_required_splits(str(out)); audit_group_integrity(splits,_group_fn); identity=build_dataset_identity(splits,include_rows=False); audit=_audit_images(out)
    archive_hash=sha256_file(src) if src.is_file() else None
    report={
        'dataset':'CFD','input':str(src),'input_archive_sha256':archive_hash,'pair_rule':'<parent>_img_<crop> <-> <parent>_msk_<crop>; canonical stem=<parent>_<crop>',
        'seed':a.seed,'requested_fractions':{'train':a.train_frac,'val':a.val_frac,'test':1-a.train_frac-a.val_frac},
        'parent_groups':{s:sorted(v) for s,v in split_parents.items()},'parent_group_counts':{s:len(v) for s,v in split_parents.items()},
        'sample_counts':{s:len(v) for s,v in splits.items()},'identity':identity,'audit':audit,'parent_group_audit':'PASS','original_data_modified':False,
    }
    (out/'CFD_PREPARATION_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
