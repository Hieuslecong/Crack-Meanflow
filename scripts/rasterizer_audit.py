from __future__ import annotations
from pathlib import Path
import json, math, os, sys
import torch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crackmeanflow.journal.geometry.rasterizer import GeometryRasterizer

def main():
    root=Path(__file__).resolve().parents[1]
    sharp=3.0;maxr=16.0
    r=GeometryRasterizer(max_radius=maxr,bins=8,sharpness=sharp,representation='centerline_edt',distance_encoding='sqrt')
    h=w=33
    center0=torch.full((1,1,h,w),.0);center1=torch.ones((1,1,h,w))
    radius=torch.zeros((1,1,h,w));radius[:,:,16,5:28]=1.0;radius[:,:,15:18,16]=2.0
    out0=r.forward_fields(center0,radius);out1=r.forward_fields(center1,radius)
    thresholds=[.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95]
    mapping=[]
    for th in thresholds:
        decoded=-math.log(1-th)/sharp
        q=math.sqrt(min(max(decoded/maxr,0),1))
        state=2*q-1
        mapping.append({'mask_threshold':th,'min_decoded_edt_px':decoded,'min_sqrt_encoded_q':q,'min_state_pm1':state})
    result={
        'representation':'centerline_edt','distance_encoding':'sqrt','sharpness':sharp,'max_radius':maxr,
        'centerline_causality_max_abs_diff':float((out0-out1).abs().max()),
        'threshold_to_decoded_field':mapping,
        'interpretation':'For centerline_edt, final mask probability is a monotonic function of predicted dense EDT only; centerline has zero direct rasterizer effect. Any OOD positive leakage in EDT can become foreground depending on the frozen threshold.'
    }
    os.makedirs('reports',exist_ok=True);json.dump(result,open('reports/RASTERIZER_AUDIT.json','w'),indent=2)
    with open('reports/RASTERIZER_AUDIT.md','w') as f:
        f.write('# Rasterizer audit\n\n')
        f.write(f"- Centerline direct-effect max |Δmask|: **{result['centerline_causality_max_abs_diff']}**\n")
        f.write('- `centerline_edt` mask is monotonic in dense EDT only.\n')
        f.write('- Threshold 0.5 corresponds to decoded EDT > %.4f px (sqrt q > %.4f; state > %.4f).\n' % (mapping[5]['min_decoded_edt_px'],mapping[5]['min_sqrt_encoded_q'],mapping[5]['min_state_pm1']))
        f.write('- This is not automatically a bug; it is a causal-design/calibration risk that must be checked on CFD/OOD predictions.\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
