"""Check public Git candidates, local links, hashes and optional private exclusions.

Supply --forbid REGEX locally for unpublished identifiers; exclusions are not
stored in the public repository. Output never prints matched secret values.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
TEXT={'.py','.ps1','.md','.txt','.json','.yaml','.yml','.toml','.tsx','.ts','.js','.mjs','.css','.svg','.html','.csv','.example'}
MANIFEST='docs/evidence/publication-manifest.json'

def main():
 p=argparse.ArgumentParser();p.add_argument('--forbid',action='append',default=[]);p.add_argument('--forbid-bytes',action='append',default=[],help='Long distinctive identifiers only; short tokens match random compressed bytes');p.add_argument('--write-manifest',action='store_true');p.add_argument('--require-pdf',action='store_true');args=p.parse_args()
 candidates=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode('utf-8').split('\0')
 candidates=sorted(set(n for n in candidates if n and (ROOT/n).is_file()))
 errors=[];warnings=[];records=[];pdf_info=None
 forbidden=[re.compile(x,re.I) for x in args.forbid]
 secrets=[re.compile(x) for x in [r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{50,}',r'AKIA[A-Z0-9]{16}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']]
 def scan(name,text):
  if any(p.search(text) for p in forbidden):errors.append(f'{name}: excluded identifier')
  if any(p.search(text) for p in secrets):errors.append(f'{name}: secret-like credential')
 for name in candidates:
  file=ROOT/name;data=file.read_bytes()
  if any(re.search(pattern,data.decode('latin-1'),re.I) for pattern in args.forbid_bytes):errors.append(f'{name}: excluded byte identifier')
  if name!=MANIFEST:records.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  scan(name,name)
  if len(data)>95*1024*1024:errors.append(f'{name}: file too large for ordinary Git publication')
  if any(part in {'.env','.venv','node_modules','__pycache__','outputs','experiments'} for part in Path(name).parts):errors.append(f'{name}: private/generated path')
  if file.suffix.lower() in TEXT or file.name in {'.gitignore','.gitattributes','.python-version'}:
   try:text=data.decode('utf-8')
   except UnicodeDecodeError:errors.append(f'{name}: expected UTF-8 text');continue
   scan(name,text)
   if file.suffix=='.md':
    for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)',text):
     target=match.group(1).strip().strip('<>');url=urlsplit(target)
     if url.scheme or target.startswith('#') or not url.path:continue
     path=(file.parent/unquote(url.path)).resolve()
     if not path.exists():errors.append(f'{name}: missing relative link {target}')
  if file.suffix.lower()=='.pdf':
   try:
    import pymupdf
    with pymupdf.open(file) as doc:
     text='\n'.join(page.get_text() for page in doc);scan(name,text);scan(name,json.dumps(doc.metadata))
     bad=[]
     for i,page in enumerate(doc):
      for word in page.get_text('words'):
       if word[0]<-1 or word[1]<-1 or word[2]>page.rect.width+1 or word[3]>page.rect.height+1:bad.append(i+1)
     if bad:errors.append(f'{name}: text extends outside pages {sorted(set(bad))}')
     if '\ufffd' in text:errors.append(f'{name}: replacement glyphs in extracted text')
     pdf_info={'pages':len(doc),'bookmarks':len(doc.get_toc()),'bytes':len(data),'text_outside_pages':len(set(bad))}
   except ImportError:
    (errors if args.require_pdf else warnings).append('Install PyMuPDF for PDF checks')
  if file.suffix.lower() in {'.tif','.tiff'}:
   import rasterio
   with rasterio.open(file) as raster:
    scan(name,json.dumps(raster.tags()))
    for band in range(1,raster.count+1):scan(name,json.dumps(raster.tags(band)))
  if file.suffix.lower() in {'.jpg','.jpeg','.png'}:
   from PIL import Image
   with Image.open(file) as image:scan(name,str(image.info));scan(name,str(dict(image.getexif())))
 # Bind hand-maintained model paths to publication identity without loading weights.
 manifest=json.loads((ROOT/'model-artifacts.json').read_text(encoding='utf-8'))
 for entry in manifest['artifacts']:
  if not re.fullmatch('[a-f0-9]{64}',entry['sha256']):errors.append('Invalid model SHA-256')
  if not entry['url'].startswith('https://github.com/GauravR17A/monocular-surface-reconstruction/releases/download/'):errors.append('Unexpected artifact origin')
 required=['README.md','docs/README.md','THIRD_PARTY_NOTICES.md','licenses/SEGFORMER.txt','licenses/CONVNEXT_MODEL.txt','docs/Monocular_Surface_Reconstruction_Technical_Documentation.pdf']
 for name in required:
  if name not in candidates:errors.append(f'Missing publication file: {name}')
 if args.write_manifest:
  (ROOT/MANIFEST).write_text(json.dumps({'status_date':'2026-09-28','scope':'Public Git candidate bytes; this manifest is excluded from its own inventory','files':records},indent=2)+'\n',encoding='utf-8')
 elif (ROOT/MANIFEST).exists():
  saved=json.loads((ROOT/MANIFEST).read_text(encoding='utf-8'))['files']
  if saved!=records:errors.append('Publication manifest differs; review changes before regenerating')
 result={'passed':not errors,'public_files':len(candidates),'total_bytes':sum(r['bytes'] for r in records),'pdf':pdf_info,'errors':errors,'warnings':warnings}
 print(json.dumps(result,indent=2));return 1 if errors else 0

if __name__=='__main__':raise SystemExit(main())
