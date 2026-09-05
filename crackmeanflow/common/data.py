"""Paired crack dataset, strict split integrity and content-level provenance."""
from __future__ import annotations
import hashlib,json,os,random
from typing import Dict,List,Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageEnhance
from torch.utils.data import Dataset,Sampler

IMG_EXTS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
Pair=Tuple[str,str,str]

def _stem_map(directory:str)->Dict[str,str]:
    out={}
    for name in os.listdir(directory):
        if os.path.splitext(name)[1].lower() not in IMG_EXTS: continue
        stem=os.path.splitext(name)[0]
        if stem in out: raise RuntimeError(f'duplicate stem {stem!r} in {directory}: {out[stem]!r} and {name!r}')
        out[stem]=name
    return out

def _list_pairs(image_dir:str,mask_dir:str)->List[Pair]:
    images=_stem_map(image_dir); masks=_stem_map(mask_dir)
    missing_masks=sorted(set(images)-set(masks)); missing_images=sorted(set(masks)-set(images))
    if missing_masks or missing_images:
        raise RuntimeError(f'unpaired files detected: missing_masks={missing_masks[:10]} ({len(missing_masks)} total), missing_images={missing_images[:10]} ({len(missing_images)} total)')
    return [(s,os.path.join(image_dir,images[s]),os.path.join(mask_dir,masks[s])) for s in sorted(images)]

def discover_pairs(root:str,split:str)->List[Pair]:
    root=os.path.abspath(root); txt=os.path.join(root,f'{split}.txt'); imgs_dir,masks_dir=os.path.join(root,'Images'),os.path.join(root,'Labels')
    if os.path.isfile(txt) and os.path.isdir(imgs_dir) and os.path.isdir(masks_dir):
        names=[line.strip().split()[0] for line in open(txt,encoding='utf-8') if line.strip()]
        if len(names)!=len(set(names)): raise RuntimeError(f'duplicate entries in split file {txt}')
        out=[]; missing=[]
        for n in names:
            stem,ext=os.path.splitext(n); candidates=[n] if ext else [n+e for e in sorted(IMG_EXTS)]; found=None
            for base in candidates:
                ip,mp=os.path.join(imgs_dir,base),os.path.join(masks_dir,base)
                if os.path.isfile(ip) and os.path.isfile(mp): found=(os.path.splitext(base)[0],ip,mp); break
            if found is None: missing.append(n)
            else: out.append(found)
        if missing: raise FileNotFoundError(f'{len(missing)} entries in {txt} are missing image/mask pairs; examples={missing[:10]}')
        return out
    imgs_dir=os.path.join(root,split,'cracked','images'); masks_dir=os.path.join(root,split,'cracked','masks')
    if os.path.isdir(imgs_dir) and os.path.isdir(masks_dir): return _list_pairs(imgs_dir,masks_dir)
    raise FileNotFoundError(f'required split={split!r} is missing under {root}')

def discover_required_splits(root:str)->Dict[str,List[Pair]]:
    result={s:discover_pairs(root,s) for s in ('train','val','test')}
    for s,pairs in result.items():
        if not pairs: raise RuntimeError(f'required split {s!r} is empty under {root}')
    audit_split_integrity(result); return result

def audit_split_integrity(splits):
    names={k:{p[0] for p in v} for k,v in splits.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        overlap=names[a]&names[b]
        if overlap: raise RuntimeError(f'split leakage {a}<->{b}: {len(overlap)} shared stems; examples={sorted(overlap)[:10]}')

def _file_sha256(path,chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()

def manifest_hash(pairs):
    payload='\n'.join(f'{n}\t{os.path.basename(i)}\t{os.path.basename(m) if m else "<EMPTY_MASK>"}' for n,i,m in sorted(pairs))
    return hashlib.sha256(payload.encode()).hexdigest()

def content_manifest_rows(pairs):
    return [
        {'name':n,'image_sha256':_file_sha256(i),'mask_sha256':_file_sha256(m) if m else '<EMPTY_MASK>'}
        for n,i,m in sorted(pairs)
    ]

def content_manifest_hash(pairs):
    rows=content_manifest_rows(pairs)
    payload='\n'.join(f"{r['name']}\t{r['image_sha256']}\t{r['mask_sha256']}" for r in rows)
    return hashlib.sha256(payload.encode()).hexdigest()

def split_identity(pairs,include_rows=False):
    out={'count':len(pairs),'name_manifest_sha256':manifest_hash(pairs),'content_manifest_sha256':content_manifest_hash(pairs)}
    if include_rows: out['files']=content_manifest_rows(pairs)
    return out

def build_dataset_identity(splits,include_rows=False):
    return {s:split_identity(p,include_rows=include_rows) for s,p in splits.items()}

def write_split_manifest(splits,path,include_content_rows=True):
    payload=build_dataset_identity(splits,include_rows=include_content_rows)
    os.makedirs(os.path.dirname(path) or '.',exist_ok=True)
    with open(path,'w',encoding='utf-8') as f: json.dump(payload,f,indent=2)
    return payload

def verify_source_provenance(checkpoint,current_splits,keys=('train','val')):
    saved=(checkpoint.get('source_provenance') or {}).get('splits') or {}
    if not saved:
        # backward compatible fallback, but content verification cannot be claimed.
        legacy=checkpoint.get('split_manifest_hashes') or {}
        for key in keys:
            if key not in legacy: raise RuntimeError(f'checkpoint has no source provenance for {key!r}')
            current=manifest_hash(current_splits[key])
            if current!=legacy[key]: raise RuntimeError(f'legacy source split mismatch for {key}: checkpoint={legacy[key]} current={current}')
        return {'verified':'name-only-legacy','keys':list(keys)}
    for key in keys:
        if key not in saved: raise RuntimeError(f'checkpoint source provenance missing split={key!r}')
        cur=split_identity(current_splits[key],include_rows=False)
        for field in ('count','name_manifest_sha256','content_manifest_sha256'):
            if saved[key].get(field)!=cur[field]:
                raise RuntimeError(f'source provenance mismatch split={key} field={field}: checkpoint={saved[key].get(field)} current={cur[field]}')
    return {'verified':'content','keys':list(keys)}

def target_dataset_identity(splits,dataset_name=None,dataset_version=None,keys=('test',)):
    return {'dataset_name':dataset_name,'dataset_version':dataset_version,'splits':{k:split_identity(splits[k],include_rows=False) for k in keys}}

# Kept for legacy callers. OOD evaluation should use target_dataset_identity instead.
def verify_checkpoint_split_provenance(checkpoint,splits,keys=('val','test')):
    saved=checkpoint.get('split_manifest_hashes') or {}
    for key in keys:
        if key not in saved: raise RuntimeError(f'checkpoint has no recorded split hash for {key!r}')
        current=manifest_hash(splits[key])
        if current!=saved[key]: raise RuntimeError(f'split provenance mismatch for {key}: checkpoint={saved[key]} current={current}')

def _mask_binary_array(mask, mode='auto_binary_safe'):
    arr=np.asarray(mask,dtype=np.float32)
    mode=str(mode)
    if mode=='threshold_127': return (arr>127).astype(np.float32)
    if mode=='nonzero': return (arr>0).astype(np.float32)
    if mode!='auto_binary_safe': raise ValueError(f'unknown mask_binarization={mode!r}')
    # Common crack masks are encoded as {0,255}, but some datasets use {0,1}.
    # Auto mode is deliberately conservative: low-range integer/binary masks use
    # nonzero semantics; conventional 8-bit masks retain the historical >127 rule.
    finite=arr[np.isfinite(arr)]
    if finite.size==0: raise RuntimeError('mask contains no finite pixels')
    mx=float(finite.max()); mn=float(finite.min())
    if mn<0: raise RuntimeError(f'mask contains negative values: min={mn}')
    if mx<=1.0: return (arr>0).astype(np.float32)
    # Values entirely in (1,127] are ambiguous: applying the historical >127
    # rule would silently erase every positive label (e.g. masks encoded {0,2}).
    if mx<=127.0:
        vals=np.unique(finite)[:20].tolist()
        raise RuntimeError(f'ambiguous low-range mask encoding with max={mx}; values={vals}. Canonicalize labels or select an explicit binarization mode.')
    return (arr>127).astype(np.float32)

def _resize_mask_pil(mask,size,mode,binarization='auto_binary_safe'):
    if mode=='nearest':
        out=mask.resize((size,size),Image.Resampling.NEAREST)
        return Image.fromarray((_mask_binary_array(out,binarization)*255).astype(np.uint8),mode='L')
    arr=torch.from_numpy(_mask_binary_array(mask,binarization))[None,None]
    if mode=='max_preserve':
        h,w=arr.shape[-2:]
        if h%size==0 and w%size==0 and h>=size and w>=size: out=F.max_pool2d(arr,kernel_size=(h//size,w//size),stride=(h//size,w//size))
        else: out=(F.interpolate(arr,size=(size,size),mode='area')>0).float()
    elif mode=='area_threshold': out=(F.interpolate(arr,size=(size,size),mode='area')>=.25).float()
    else: raise ValueError(f'unknown mask_resize_mode={mode!r}')
    return Image.fromarray((out[0,0].numpy()*255).astype(np.uint8),mode='L')

class PairedCrackDataset(Dataset):
    def __init__(self,pairs,image_size=256,augment=False,photometric=False,mask_resize_mode='nearest',mask_binarization='auto_binary_safe'):
        self.pairs=list(pairs); self.size=int(image_size); self.augment=bool(augment); self.photometric=bool(photometric); self.mask_resize_mode=str(mask_resize_mode); self.mask_binarization=str(mask_binarization)
    def __len__(self): return len(self.pairs)
    def __getitem__(self,idx):
        name,ip,mp=self.pairs[idx]
        with Image.open(ip) as _im:
            img=_im.convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR)
        if mp is None:
            msk=Image.fromarray(np.zeros((self.size,self.size),dtype=np.uint8),mode='L')
        else:
            with Image.open(mp) as _msk:
                msk=_resize_mask_pil(_msk.convert('L'),self.size,self.mask_resize_mode,self.mask_binarization)
        if self.augment:
            if random.random()<.5: img,msk=img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),msk.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random()<.5: img,msk=img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),msk.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if random.random()<.5:
                trans={1:Image.Transpose.ROTATE_90,2:Image.Transpose.ROTATE_180,3:Image.Transpose.ROTATE_270}[random.choice([1,2,3])]
                img,msk=img.transpose(trans),msk.transpose(trans)
            if self.photometric:
                img=ImageEnhance.Brightness(img).enhance(random.uniform(.8,1.2)); img=ImageEnhance.Contrast(img).enhance(random.uniform(.8,1.2)); img=ImageEnhance.Color(img).enhance(random.uniform(.8,1.2))
        img_t=torch.from_numpy(np.asarray(img,dtype=np.float32)/255.).permute(2,0,1); mask01=torch.from_numpy((np.asarray(msk)>127).astype(np.float32))[None]
        return {'name':name,'crack':img_t,'mask':mask01*2.-1.}

def audit_group_integrity(splits,group_fn):
    groups={k:{str(group_fn(p[0])) for p in v} for k,v in splits.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        overlap=groups[a]&groups[b]
        if overlap: raise RuntimeError(f'group leakage {a}<->{b}: {len(overlap)} shared groups; examples={sorted(overlap)[:10]}')

def discover_normal_images(root,split):
    d=os.path.join(os.path.abspath(root),split,'normal','images')
    if not os.path.isdir(d): return []
    fmap=_stem_map(d); return [(f'NORMAL::{stem}',os.path.join(d,name),None) for stem,name in sorted(fmap.items())]

def append_normal_negatives(pairs,normals):
    out=list(pairs); existing={p[0] for p in out}
    for p in normals:
        if p[0] in existing: raise RuntimeError(f'normal-negative name collision: {p[0]!r}')
        out.append(p)
    return out

def source_id_from_stem(stem,pattern=r'^([^_]+)'):
    import re
    if stem.startswith('NORMAL::'): return 'NORMAL'
    m=re.search(pattern,stem)
    if not m: raise RuntimeError(f'source regex {pattern!r} did not match stem {stem!r}')
    return m.group(1) if m.groups() else m.group(0)

class EpochRandomSampler(Sampler):
    """Deterministic no-replacement epoch sampler.

    Unlike DataLoader(shuffle=True), the ordering of a resumed epoch depends only
    on (seed, epoch), not on how many previous DataLoader iterators were created.
    This makes epoch-boundary resume reproducible without replacement sampling.
    """
    def __init__(self,num_samples,seed=0):
        self.num_samples=int(num_samples); self.seed=int(seed); self.epoch=0
        if self.num_samples < 1: raise ValueError('num_samples must be >=1')
    def set_epoch(self,epoch): self.epoch=int(epoch)
    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed+self.epoch)
        return iter(torch.randperm(self.num_samples,generator=g).tolist())
    def __len__(self): return self.num_samples

class EpochWeightedRandomSampler(Sampler):
    def __init__(self,weights,num_samples,replacement=True,seed=0):
        self.weights=torch.as_tensor(weights,dtype=torch.double); self.num_samples=int(num_samples); self.replacement=bool(replacement); self.seed=int(seed); self.epoch=0
    def set_epoch(self,epoch): self.epoch=int(epoch)
    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed+self.epoch); return iter(torch.multinomial(self.weights,self.num_samples,self.replacement,generator=g).tolist())
    def __len__(self): return self.num_samples

def source_balancing_weights(pairs,pattern=r'^([^_]+)',power=.5,cap_ratio=4.):
    from collections import Counter
    src=[source_id_from_stem(p[0],pattern) for p in pairs]; cnt=Counter(src); w=np.asarray([cnt[s]**(-float(power)) for s in src],dtype=np.float64); med=float(np.median(w)) if len(w) else 1.
    if cap_ratio and cap_ratio>0: w=np.minimum(w,med*float(cap_ratio))
    return w.tolist(),dict(cnt)

def discover_evaluation_pairs(root,split='test',include_normal_negatives=False):
    """Discover a benchmark split, optionally including explicit normal/no-mask images.

    Normal images are represented with ``mask_path=None`` and therefore evaluate
    against an all-background target. This prevents silent omission of negative
    examples from OOD benchmarks such as mixed crack/non-crack collections.
    """
    pairs=list(discover_pairs(root,split))
    if include_normal_negatives:
        pairs=append_normal_negatives(pairs,discover_normal_images(root,split))
    return pairs

def audit_mask_encoding(pairs,max_files=None):
    """Inspect mask encoding without changing dataset content.

    Returns enough information to detect a dangerous 0/1-vs-0/255 convention
    mismatch before evaluation. ``None`` masks (explicit normal negatives) are
    counted separately.
    """
    values=set(); dims={}; missing_masks=0; positives=0; empty_explicit=0; inspected=0
    iterable=pairs if max_files is None else pairs[:int(max_files)]
    for _,ip,mp in iterable:
        with Image.open(ip) as im:
            dims[tuple(im.size)]=dims.get(tuple(im.size),0)+1
        if mp is None:
            missing_masks+=1; inspected+=1; continue
        with Image.open(mp).convert('L') as m:
            a=np.asarray(m)
            values.update(int(v) for v in np.unique(a))
            has_pos=bool((a>0).any()); positives+=int(has_pos); empty_explicit+=int(not has_pos)
        inspected+=1
    vals=sorted(values)
    if not vals:
        convention='empty-mask-files-only' if missing_masks else 'unknown'
    elif max(vals)<=1:
        convention='binary_0_1'
    elif set(vals).issubset({0,255}):
        convention='binary_0_255'
    else:
        convention='grayscale_or_multiclass'
    return {
        'inspected_pairs':inspected,
        'explicit_normal_no_mask_count':missing_masks,
        'mask_unique_values':vals,
        'mask_encoding_convention':convention,
        'nonempty_mask_files':positives,
        'empty_explicit_mask_files':empty_explicit,
        'image_dimensions':{f'{w}x{h}':n for (w,h),n in sorted(dims.items())},
    }

def verify_source_dataset_contract(checkpoint,current_splits,dataset_name,dataset_version,keys=('train','val','test')):
    saved=checkpoint.get('source_provenance') or {}
    if not saved:
        return verify_source_provenance(checkpoint,current_splits,keys=keys)
    if saved.get('dataset_name')!=dataset_name:
        raise RuntimeError(f'source dataset name mismatch: checkpoint={saved.get("dataset_name")!r} current={dataset_name!r}')
    if saved.get('dataset_version')!=dataset_version:
        raise RuntimeError(f'source dataset version mismatch: checkpoint={saved.get("dataset_version")!r} current={dataset_version!r}')
    result=verify_source_provenance(checkpoint,current_splits,keys=keys)
    result.update({'dataset_name':dataset_name,'dataset_version':dataset_version})
    return result

def load_binary_mask_native(mask_path,binarization='auto_binary_safe',fallback_hw=None):
    if mask_path is None:
        if fallback_hw is None: raise ValueError('fallback_hw is required for an explicit normal image without a mask file')
        return torch.zeros(1,int(fallback_hw[0]),int(fallback_hw[1]),dtype=torch.float32)
    with Image.open(mask_path).convert('L') as m:
        arr=_mask_binary_array(m,binarization)
    return torch.from_numpy(arr)[None].float()


def audit_within_split_duplicate_images(splits):
    """Fail on renamed exact duplicate image bytes within a source split.

    Repeated mask bytes are common for empty/simple masks and are not by themselves
    evidence of duplicated observations; image-content duplicates are the strict key.
    """
    duplicates={}
    for split,pairs in splits.items():
        owners={}; groups={}
        for name,ip,_ in pairs:
            h=_file_sha256(ip); owners.setdefault(h,[]).append(name)
        groups={h:names for h,names in owners.items() if len(names)>1}
        if groups: duplicates[split]=groups
    if duplicates:
        examples={k:list(v.items())[:5] for k,v in duplicates.items()}
        raise RuntimeError(f'exact duplicate image-content within source split(s): {examples}')
    return {'within_split_exact_duplicate_image_groups':0}

def audit_content_split_integrity(splits):
    """Fail if exact image or mask bytes appear in more than one source split.

    Stem/group checks are insufficient when duplicated files have been renamed.
    Scientific source splits must therefore also be disjoint at byte-content level.
    Explicit normal negatives have no mask and are checked by image content only.
    """
    image_owner={}
    mask_owner={}
    image_overlaps=[]
    mask_overlaps=[]
    for split,pairs in splits.items():
        for name,ip,mp in pairs:
            ih=_file_sha256(ip)
            prev=image_owner.get(ih)
            if prev is not None and prev[0]!=split:
                image_overlaps.append({'sha256':ih,'a':prev,'b':(split,name)})
            else:
                image_owner[ih]=(split,name)
            if mp is not None:
                mh=_file_sha256(mp)
                prevm=mask_owner.get(mh)
                if prevm is not None and prevm[0]!=split:
                    mask_overlaps.append({'sha256':mh,'a':prevm,'b':(split,name)})
                else:
                    mask_owner[mh]=(split,name)
    if image_overlaps:
        raise RuntimeError(f'exact image-content leakage across source splits: {len(image_overlaps)} overlaps; examples={image_overlaps[:5]}')
    # Identical empty/background masks can legitimately recur, so mask-content
    # equality alone is not a leakage criterion. We return it diagnostically.
    return {'image_content_overlap_count':0,'mask_content_cross_split_duplicate_count':len(mask_overlaps),'mask_content_examples':mask_overlaps[:20]}

def source_splits_for_config(root,cfg):
    """Reconstruct the exact source split composition implied by a config.

    This is shared by training, resume verification, threshold freezing and
    evaluation so optional source normal negatives cannot silently drift.
    """
    base=discover_required_splits(root)
    nc=((cfg or {}).get('train') or {}).get('normal_negatives') or {}
    out={k:list(v) for k,v in base.items()}
    for split in ('train','val','test'):
        if bool(nc.get(split,False)):
            normals=discover_normal_images(root,split)
            if not normals:
                raise RuntimeError(f'config requires normal negatives for split={split!r}, but no {split}/normal/images were found')
            out[split]=append_normal_negatives(out[split],normals)
    audit_split_integrity(out)
    audit_content_split_integrity(out)
    audit_within_split_duplicate_images(out)
    return out


def audit_cross_dataset_image_content_overlap(source_splits,target_pairs):
    """Fail fast on exact image-byte overlap between source splits and a target benchmark."""
    from collections import defaultdict
    source_hashes=defaultdict(list)
    for split,pairs in source_splits.items():
        for name,image_path,_ in pairs:
            source_hashes[_file_sha256(image_path)].append((str(split),str(name)))
    overlap=[]
    for name,image_path,_ in target_pairs:
        h=_file_sha256(image_path)
        if h in source_hashes:
            overlap.append({'target_name':str(name),'image_sha256':h,'source_matches':source_hashes[h]})
    if overlap:
        raise RuntimeError(f'exact source-target image-content contamination: {len(overlap)} target images overlap source; examples={overlap[:5]}')
    return {'exact_source_target_image_content_overlap_count':0,'source_image_hash_count':len(source_hashes),'target_count':len(target_pairs)}
