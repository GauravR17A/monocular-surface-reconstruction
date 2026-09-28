"""Render a selected archived urban result without changing prediction values."""
from pathlib import Path
import json, hashlib
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
JOB=ROOT/'outputs/web_jobs/1e2368ead1ee4777a02366a3212705ab'
OUT=ROOT/'outputs/reports/selected_urban_20260916';OUT.mkdir(parents=True,exist_ok=True)
def read(p):
    with rasterio.open(p) as d:return d.read(1).astype(np.float64)
pr=read(JOB/'height_above_ground_m.tif')
ref_path=JOB/'validation_reference.tif'
if (JOB/'validation_reference_aligned_m.tif').exists():ref_path=JOB/'validation_reference_aligned_m.tif'
gt=read(ref_path)
assert pr.shape==gt.shape
m=np.isfinite(gt)&np.isfinite(pr)&(gt>2)
e=pr[m]-gt[m]
info={'source_job':str(JOB),'selection':'Purposively selected existing urban example; not representative performance.',
      'reference':'Supplied building-height reference; LiDAR origin not authenticated. Zeros not treated as valid ground.',
      'score_scope':'Finite supplied reference pixels greater than 2 m; not a measured-only building benchmark.',
      'pixel_count':int(m.sum()),'fraction':float(m.mean()),'rmse_m':float(np.sqrt(np.mean(e**2))),
      'mae_m':float(np.mean(abs(e))),'prediction_sha256':hashlib.sha256((JOB/'height_above_ground_m.tif').read_bytes()).hexdigest(),
      'metadata':json.loads((JOB/'metadata.json').read_text())}
vmax=float(np.ceil(max(np.nanmax(pr),np.nanmax(gt))))
cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#e5e9ec')
fig,axes=plt.subplots(1,3,figsize=(15,6.15),gridspec_kw={'wspace':.055})
fig.patch.set_facecolor('white')
axes[0].imshow(Image.open(JOB/'texture.jpg'))
im=axes[1].imshow(pr,cmap=cmap,vmin=0,vmax=vmax,interpolation='nearest')
# Zero labels cannot safely be interpreted as measured ground in this uploaded reference.
axes[2].imshow(np.ma.masked_where(~np.isfinite(gt)|(gt<=0),gt),cmap=cmap,vmin=0,vmax=vmax,interpolation='nearest')
for ax,t in zip(axes,['Optical RGB input','Raw predicted height','Supplied height reference']):
    ax.set_title(t,fontsize=16,color='#18363D',pad=11);ax.axis('off')
fig.subplots_adjust(left=.015,right=.985,top=.93,bottom=.19)
cbax=fig.add_axes([.25,.085,.5,.025]);fig.colorbar(im,cax=cbax,orientation='horizontal').set_label('Height above ground (m) · shared colour scale',fontsize=11)
fig.savefig(OUT/'urban_selected_comparison.png',dpi=160,facecolor='white');plt.close(fig)
(OUT/'selection_audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
print(json.dumps({k:info[k]for k in ['pixel_count','fraction','rmse_m','mae_m']},indent=2))
