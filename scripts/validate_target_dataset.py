from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crackmeanflow.common import discover_evaluation_pairs, discover_normal_images, split_identity, audit_mask_encoding, content_manifest_rows, TARGET_LOCK_TYPE, source_tree_hash

def main():
    ap=argparse.ArgumentParser(description='Create a strict immutable OOD test identity lock before any headline evaluation.')
    ap.add_argument('--data',required=True); ap.add_argument('--dataset-name',required=True); ap.add_argument('--dataset-version',required=True)
    ap.add_argument('--include-normal-negatives',action='store_true',help='include test/normal/images as explicit empty-mask negatives')
    ap.add_argument('--exclude-normal-negatives-with-justification',default=None,help='required if normal images exist but the canonical benchmark intentionally excludes them')
    ap.add_argument('--allow-empty-explicit-masks-with-justification',default=None,help='required if files under test/cracked/masks are empty by benchmark design')
    ap.add_argument('--allow-exact-duplicate-images',action='store_true'); ap.add_argument('--duplicate-justification',default=None)
    ap.add_argument('--expected-count',type=int,default=None); ap.add_argument('--expected-content-sha256',default=None); ap.add_argument('--benchmark-scope',required=True,help='human-readable exact benchmark scope, e.g. official full test set + inclusion policy')
    ap.add_argument('--out',required=True); a=ap.parse_args()
    if not str(a.dataset_name).strip() or not str(a.dataset_version).strip(): raise ValueError('target dataset name/version must be non-empty')
    if not str(a.benchmark_scope).strip(): raise ValueError('--benchmark-scope must be non-empty')
    existing_normals=discover_normal_images(a.data,'test')
    if existing_normals and not a.include_normal_negatives and not a.exclude_normal_negatives_with_justification:
        raise RuntimeError(f'{len(existing_normals)} normal-negative images exist but would be silently excluded; either --include-normal-negatives or provide --exclude-normal-negatives-with-justification')
    if a.include_normal_negatives and a.exclude_normal_negatives_with_justification:
        raise RuntimeError('cannot both include normal negatives and justify excluding them')
    pairs=discover_evaluation_pairs(a.data,'test',a.include_normal_negatives)
    if not pairs: raise RuntimeError('target test split is empty')
    ident=split_identity(pairs,include_rows=False); encoding=audit_mask_encoding(pairs); checks={'count_match':None,'content_hash_match':None}
    rows=content_manifest_rows(pairs); by_image={}
    for r in rows: by_image.setdefault(r['image_sha256'],[]).append(r['name'])
    dup_images={h:n for h,n in by_image.items() if len(n)>1}
    if dup_images and not a.allow_exact_duplicate_images: raise RuntimeError(f'target benchmark contains {len(dup_images)} exact duplicate image-content groups; canonicalize or explicitly justify')
    if dup_images and a.allow_exact_duplicate_images and not a.duplicate_justification: raise RuntimeError('--allow-exact-duplicate-images requires --duplicate-justification')
    if encoding['empty_explicit_mask_files'] and not a.allow_empty_explicit_masks_with_justification:
        raise RuntimeError(f'{encoding["empty_explicit_mask_files"]} explicit masks under cracked split are empty; canonicalize or provide --allow-empty-explicit-masks-with-justification')
    if a.expected_count is not None:
        checks['count_match']=ident['count']==a.expected_count
        if not checks['count_match']: raise RuntimeError(f'target count mismatch: expected={a.expected_count} actual={ident["count"]}')
    if a.expected_content_sha256:
        checks['content_hash_match']=ident['content_manifest_sha256']==a.expected_content_sha256
        if not checks['content_hash_match']: raise RuntimeError('target content hash mismatch')
    if encoding['mask_encoding_convention']=='grayscale_or_multiclass':
        raise RuntimeError(f'ambiguous/multivalued mask encoding {encoding["mask_unique_values"][:20]}; canonicalize labels before paper evaluation')
    verification_level='EXTERNALLY_MATCHED_IDENTITY' if a.expected_count is not None and a.expected_content_sha256 else ('COUNT_MATCHED_IDENTITY' if a.expected_count is not None else 'LOCALLY_FROZEN_IDENTITY')
    report={'lock_type':TARGET_LOCK_TYPE,'dataset_name':a.dataset_name,'dataset_version':a.dataset_version,'benchmark_scope':a.benchmark_scope,
            'include_normal_negatives':bool(a.include_normal_negatives),'normal_images_present':len(existing_normals),'normal_exclusion_justification':a.exclude_normal_negatives_with_justification,
            'test_identity':ident,'mask_audit':encoding,'empty_mask_justification':a.allow_empty_explicit_masks_with_justification,
            'exact_duplicate_image_content_groups':len(dup_images),'exact_duplicate_image_examples':list(dup_images.items())[:20],'duplicate_justification':a.duplicate_justification,
            'checks':checks,'identity_verification_level':verification_level,'validator_source_tree_sha256':source_tree_hash(),'status':'PASS'}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__': main()
