from __future__ import annotations
from pathlib import Path
import argparse, json, os, sys
import torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crackmeanflow.journal.flow.improved_meanflow import _stratified_mask


def sample_a5(n:int, seed:int, mu:float, sigma:float, data_proportion:float):
    g=torch.Generator(device='cpu').manual_seed(seed)
    # Chunk to keep memory bounded and preserve the exact distribution.
    chunk=100_000
    counts={'n':0,'exact':0,'near_90_10':0,'near_95_05':0,'gap_ge_08':0,'gap_ge_09':0,'fm':0}
    tsum=rsum=gapsum=0.0
    offset=0
    while offset<n:
        b=min(chunk,n-offset)
        z=torch.randn((2,b),generator=g)*sigma+mu
        a,bb=torch.sigmoid(z[0]),torch.sigmoid(z[1])
        t=torch.maximum(a,bb);r=torch.minimum(a,bb)
        fm=_stratified_mask(b,data_proportion,offset,'cpu')
        r=torch.where(fm,t,r)
        gap=t-r
        counts['n']+=b
        counts['exact']+=int(((t==1)&(r==0)).sum())
        counts['near_90_10']+=int(((t>=.9)&(r<=.1)).sum())
        counts['near_95_05']+=int(((t>=.95)&(r<=.05)).sum())
        counts['gap_ge_08']+=int((gap>=.8).sum())
        counts['gap_ge_09']+=int((gap>=.9).sum())
        counts['fm']+=int(fm.sum())
        tsum+=float(t.sum());rsum+=float(r.sum());gapsum+=float(gap.sum())
        offset+=b
    out={
        'samples':n,
        'seed':seed,
        't_mean':tsum/n,
        'r_mean':rsum/n,
        'gap_mean':gapsum/n,
        'fm_fraction':counts['fm']/n,
        'exact_deployment_fraction':counts['exact']/n,
        'near_t>=0.90_r<=0.10_fraction':counts['near_90_10']/n,
        'near_t>=0.95_r<=0.05_fraction':counts['near_95_05']/n,
        'gap>=0.80_fraction':counts['gap_ge_08']/n,
        'gap>=0.90_fraction':counts['gap_ge_09']/n,
    }
    return out


def main():
    root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--a5-config',default=str(root/'configs/journal/a5_geocrack_imf_baseline.yaml'));ap.add_argument('--conference-config',default=str(root/'configs/conference/crackmeanflow_unet.yaml'));ap.add_argument('--samples',type=int,default=2_000_000);ap.add_argument('--seed',type=int,default=123);ap.add_argument('--out',default=str(root/'reports/ENDPOINT_MISMATCH_AUDIT.json'));a=ap.parse_args()
    a5=yaml.safe_load(open(a.a5_config));conf=yaml.safe_load(open(a.conference_config))
    l=a5['loss']; a5_stats=sample_a5(a.samples,a.seed,float(l.get('time_mu',-.4)),float(l.get('time_sigma',1.)),float(l.get('data_proportion',.5)))
    final_stage=(conf.get('loss',{}).get('time_curriculum',{}).get('stages') or [])[-1]
    conference_endpoint=float(final_stage.get('boundary_prob',conf.get('loss',{}).get('boundary_prob',0.)))
    result={
        'A5_current_sampling':a5_stats,
        'Conference_final_curriculum_explicit_endpoint_probability':conference_endpoint,
        'deployment_contract':{'r':0.0,'t':1.0,'nfe':1,'initial_state':'pure_noise'},
        'verdict':'CONFIRMED_TRAIN_DEPLOY_ENDPOINT_MISMATCH' if a5_stats['near_t>=0.90_r<=0.10_fraction']<0.001 and conference_endpoint>=.10 else 'REVIEW',
    }
    os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True);json.dump(result,open(a.out,'w'),indent=2)
    md=os.path.splitext(a.out)[0]+'.md'
    with open(md,'w') as f:
        f.write('# Endpoint mismatch audit\n\n')
        f.write(f"**Verdict: {result['verdict']}**\n\n")
        f.write(f"- A5 samples: {a.samples:,}\n")
        f.write(f"- A5 exact `(t=1,r=0)`: {a5_stats['exact_deployment_fraction']:.8f}\n")
        f.write(f"- A5 near `(t>=0.90,r<=0.10)`: {a5_stats['near_t>=0.90_r<=0.10_fraction']:.8f}\n")
        f.write(f"- A5 near `(t>=0.95,r<=0.05)`: {a5_stats['near_t>=0.95_r<=0.05_fraction']:.8f}\n")
        f.write(f"- Conference final explicit endpoint probability: {conference_endpoint:.4f}\n")
        f.write('- Deployment: pure noise, `(r,t)=(0,1)`, NFE=1.\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
