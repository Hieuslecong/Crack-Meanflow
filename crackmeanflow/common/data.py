"""Shared paired crack dataset and split integrity helpers."""
from __future__ import annotations
import hashlib,json,os,random
from typing import Dict,Iterable,List,Tuple
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image,ImageEnhance
from torch.utils.data import Dataset,Sampler
IMG_EXTS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}; Pair=Tuple[str,str,str]
def _stem_map(directory):
    out={}
    for name in os.listdir(directory):
        if os.path.splitext(name)[1].lower() not in IMG_EXTS: continue
        stem=os.path.splitext(name)[0]
        if stem in out: raise RuntimeError(f'duplicate stem {stem!r} in {directory}')
        out[stem]=name
    return out
def _list_pairs(image_dir,mask_dir):
    images=_stem_map(image_dir); masks=_stem_map(mask_dir); mm=sorted(set(images)-set(masks)); mi=sorted(set(masks)-set(images))
    if mm or mi: raise RuntimeError(f'unpaired files: missing_masks={mm[:10]}, missing_images={mi[:10]}')
    return [(s,os.path.join(image_dir,images[s]),os.path.join(mask_dir,masks[s])) for s in sorted(images)]
def discover_pairs(root,split):
    root=os.path.abspath(root); txt=os.path.join(root,f'{split}.txt'); imgs_dir=os.path.join(root,'Images'); masks_dir=os.path.join(root,'Labels')
    if os.path.isfile(txt) and os.path.isdir(imgs_dir) and os.path.isdir(masks_dir):
        names=[line.strip().split()[0] for line in open(txt,encoding='utf-8') if line.strip()]; out=[]; missing=[]
        if len(names)!=len(set(names)): raise RuntimeError(f'duplicate entries in {txt}')
        for n in names:
            stem,ext=os.path.splitext(n); candidates=[n] if ext else [n+e for e in sorted(IMG_EXTS)]; found=None
            for base in candidates:
                ip,mp=os.path.join(imgs_dir,base),os.path.join(masks_dir,base)
                if os.path.isfile(ip) and os.path.isfile(mp): found=(os.path.splitext(base)[0],ip,mp); break
            if found is None: missing.append(n)
            else: out.append(found)
        if missing: raise FileNotFoundError(f'{len(missing)} missing pairs in {txt}: {missing[:10]}')
        return out
    i=os.path.join(root,split,'cracked','images'); m=os.path.join(root,split,'cracked','masks')
    if os.path.isdir(i) and os.path.isdir(m): return _list_pairs(i,m)
    raise FileNotFoundError(f'required split={split!r} missing under {root}')
def discover_required_splits(root):
    result={s:discover_pairs(root,s) for s in ('train','val','test')}; audit_split_integrity(result)
    for s,p in result.items():
        if not p: raise RuntimeError(f'required split {s!r} empty')
    return result
def audit_split_integrity(splits):
    names={k:{p[0] for p in v} for k,v in splits.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        ov=names[a]&names[b]
        if ov: raise RuntimeError(f'split leakage {a}<->{b}: {len(ov)}')
def audit_group_integrity(splits,group_fn):
    groups={k:{str(group_fn(p[0])) for p in v} for k,v in splits.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        ov=groups[a]&groups[b]
        if ov: raise RuntimeError(f'group leakage {a}<->{b}: {len(ov)}')
def manifest_hash(pairs):
    payload='\n'.join(f'{n}\t{os.path.basename(i)}\t{os.path.basename(m) if m else "<EMPTY_MASK>"}' for n,i,m in sorted(pairs)); return hashlib.sha256(payload.encode()).hexdigest()
def content_manifest_hash(pairs):
    def h(p):
        x=hashlib.sha256(); x.update(open(p,'rb').read()); return x.hexdigest()
    return hashlib.sha256('\n'.join(f'{n}\t{h(i)}\t{h(m) if m else "<EMPTY_MASK>"}' for n,i,m in sorted(pairs)).encode()).hexdigest()
def write_split_manifest(splits,path):
    payload={s:{'count':len(p),'sha256':manifest_hash(p),'names':[x[0] for x in p]} for s,p in splits.items()}; os.makedirs(os.path.dirname(path) or '.',exist_ok=True); json.dump(payload,open(path,'w'),indent=2); return payload
def _resize_mask_pil(mask,size,mode):
    if mode=='nearest': return mask.resize((size,size),Image.Resampling.NEAREST)
    arr=torch.from_numpy((np.asarray(mask,dtype=np.float32)>127).astype(np.float32))[None,None]
    if mode=='max_preserve': out=(F.interpolate(arr,size=(size,size),mode='area')>0).float()
    elif mode=='area_threshold': out=(F.interpolate(arr,size=(size,size),mode='area')>=.25).float()
    else: raise ValueError(f'unknown mask_resize_mode={mode!r}')
    return Image.fromarray((out[0,0].numpy()*255).astype(np.uint8),mode='L')
class PairedCrackDataset(Dataset):
    def __init__(self,pairs,image_size=256,augment=False,photometric=False,mask_resize_mode='nearest'): self.pairs=list(pairs); self.size=int(image_size); self.augment=bool(augment); self.photometric=bool(photometric); self.mask_resize_mode=str(mask_resize_mode)
    def __len__(self): return len(self.pairs)
    def __getitem__(self,idx):
        name,ip,mp=self.pairs[idx]; img=Image.open(ip).convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR); msk=Image.fromarray(np.zeros((self.size,self.size),dtype=np.uint8),mode='L') if mp is None else _resize_mask_pil(Image.open(mp).convert('L'),self.size,self.mask_resize_mode)
        if self.augment:
            if random.random()<.5: img,msk=img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),msk.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random()<.5: img,msk=img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),msk.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if self.photometric: img=ImageEnhance.Brightness(img).enhance(random.uniform(.8,1.2)); img=ImageEnhance.Contrast(img).enhance(random.uniform(.8,1.2))
        img_t=torch.from_numpy(np.asarray(img,dtype=np.float32)/255.).permute(2,0,1); mask01=torch.from_numpy((np.asarray(msk)>127).astype(np.float32))[None]; return {'name':name,'crack':img_t,'mask':mask01*2.-1.}
def verify_checkpoint_split_provenance(checkpoint,splits,keys=('val','test')):
    saved=checkpoint.get('split_manifest_hashes') or {}
    for key in keys:
        if key not in saved or manifest_hash(splits[key])!=saved[key]: raise RuntimeError(f'split provenance mismatch for {key}')
def discover_normal_images(root,split):
    d=os.path.join(os.path.abspath(root),split,'normal','images')
    if not os.path.isdir(d): return []
    return [(f'NORMAL::{s}',os.path.join(d,n),None) for s,n in sorted(_stem_map(d).items())]
def append_normal_negatives(pairs,normals): return list(pairs)+list(normals)
def source_id_from_stem(stem,pattern=r'^([^_]+)'):
    import re
    if stem.startswith('NORMAL::'): return 'NORMAL'
    m=re.search(pattern,stem)
    if not m: raise RuntimeError(f'source regex did not match {stem!r}')
    return m.group(1) if m.groups() else m.group(0)
class EpochWeightedRandomSampler(Sampler):
    def __init__(self,weights,num_samples,replacement=True,seed=0): self.weights=torch.as_tensor(weights,dtype=torch.double); self.num_samples=int(num_samples); self.replacement=bool(replacement); self.seed=int(seed); self.epoch=0
    def set_epoch(self,epoch): self.epoch=int(epoch)
    def __iter__(self): return iter(torch.multinomial(self.weights,self.num_samples,self.replacement,generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist())
    def __len__(self): return self.num_samples
def source_balancing_weights(pairs,pattern=r'^([^_]+)',power=.5,cap_ratio=4.):
    from collections import Counter
    src=[source_id_from_stem(p[0],pattern) for p in pairs]; cnt=Counter(src); w=np.asarray([cnt[s]**(-float(power)) for s in src]); med=float(np.median(w)) if len(w) else 1.; w=np.minimum(w,med*float(cap_ratio)) if cap_ratio else w; return w.tolist(),dict(cnt)
