from __future__ import annotations
import argparse,json,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crackmeanflow.common import source_tree_hash

def main():
    ap=argparse.ArgumentParser(description='Fail-fast release/push preflight. This does not push anything.')
    ap.add_argument('--out',default='reports/RELEASE_PREFLIGHT.json');a=ap.parse_args();root=Path(__file__).resolve().parents[1]
    prov=json.loads((root/'third_party/CONFERENCE_UNET_PROVENANCE.json').read_text());issues=[]
    if prov.get('redistribution_license_status')!='CONFIRMED':issues.append('Conference U-Net redistribution rights are not confirmed; update third_party provenance with evidence before public release')
    # Absolute local paths are prohibited in source/configs intended for release.
    bad=[];rx=re.compile(r'/home/|/Users/|[A-Za-z]:\\\\')
    for base in (root/'crackmeanflow',root/'scripts',root/'configs'):
        for p in base.rglob('*'):
            if p.is_file() and p.suffix in {'.py','.yaml','.yml','.json'}:
                if p.resolve() == Path(__file__).resolve():
                    continue
                try:t=p.read_text()
                except UnicodeDecodeError:continue
                if rx.search(t):bad.append(str(p.relative_to(root)))
    if bad:issues.append(f'absolute machine-specific paths found: {bad[:20]}')
    compile_run=subprocess.run([sys.executable,'-m','compileall','-q','crackmeanflow','scripts','tests'],cwd=root,capture_output=True,text=True);compile_ok=compile_run.returncode==0
    if not compile_ok:issues.append('compileall failed')
    test=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=root,capture_output=True,text=True);tests_ok=test.returncode==0
    if not tests_ok:issues.append('pytest failed')
    protocol=subprocess.run([sys.executable,'scripts/protocol_preflight.py','--out','reports/RELEASE_PROTOCOL_PREFLIGHT.json'],cwd=root,capture_output=True,text=True);protocol_ok=protocol.returncode==0
    if not protocol_ok:issues.append('protocol preflight failed')
    report={'source_tree_sha256':source_tree_hash(),'compileall_pass':compile_ok,'tests_pass':tests_ok,'protocol_preflight_pass':protocol_ok,'third_party_provenance':prov,'absolute_path_scan_pass':not bad,'issues':issues,'release_ready':not issues,'note':'This preflight never performs a GitHub write/push.'}
    out=root/a.out if not Path(a.out).is_absolute() else Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if issues:raise SystemExit(2)
if __name__=='__main__':main()
