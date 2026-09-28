"""Render a fresh LiDAR comparison and retain the complete selection audit."""
from pathlib import Path
import json, hashlib
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reports/lidar_recheck_20260916'
audit=json.loads((OUT/'audit.json').read_text())
for row in audit['candidates']:
    row['metrics']=json.loads((Path(row['output'])/'validation_metrics.json').read_text())
rows=sorted(audit['candidates'],key=lambda r:r['metrics']['r2'],reverse=True)
best=next(row for row in rows if row['sample']=='block_0_2934');job=Path(best['output'])
def read(path):
    with rasterio.open(path) as ds:
        return ds.read(1).astype(float), ds.read_masks(1)>0, (ds.shape,ds.crs,ds.transform)
pred,pm,pg=read(job/'height_above_ground_m.tif')
gt,gm,gg=read(job/'validation_reference_aligned_m.tif')
assert pg==gg
valid=pm&gm&np.isfinite(pred)&np.isfinite(gt)
raised=valid&(gt>2)
err=pred[raised]-gt[raised]
audit['selected']=best['sample']
audit['selection_rule']='Different urban scene explicitly selected for its large building and visible roof geometry after reviewing 15 candidates. Not the best numeric result and not a blind test.'
audit['raised_surface_diagnostic']={'definition':'Reference height >2 m; buildings and trees, not building-only.',
    'rmse_m':float(np.sqrt(np.mean(err**2))),'mae_m':float(np.mean(abs(err))), 'pixel_count':int(raised.sum())}
audit['reference_source']='DFC19/US3D derivative, local SynRS3D/JasonXF prepared tiles; airborne LiDAR-derived above-ground heights.'
audit['official_source']='https://www.grss-ieee.org/community/technical-committees/2019-ieee-grss-data-fusion-contest/'
audit['prediction_sha256']=hashlib.sha256((job/'height_above_ground_m.tif').read_bytes()).hexdigest()
(OUT/'different_urban_audit.json').write_text(json.dumps(audit,indent=2))
vmax=float(np.ceil(max(pred[valid].max(),gt[valid].max())))
vmin=float(min(0,np.floor(min(pred[valid].min(),gt[valid].min()))))
fig,axes=plt.subplots(1,3,figsize=(15,6.15),gridspec_kw={'wspace':.055})
axes[0].imshow(Image.open(job/'texture.jpg'))
cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#e5e9ec')
im=axes[1].imshow(np.ma.masked_where(~valid,pred),cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
axes[2].imshow(np.ma.masked_where(~valid,gt),cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
for ax,title in zip(axes,['Satellite RGB input','Raw predicted height','LiDAR-derived reference']):
    ax.set_title(title,fontsize=16,color='#18363D',pad=11);ax.axis('off')
fig.subplots_adjust(left=.015,right=.985,top=.93,bottom=.19)
cbax=fig.add_axes([.25,.085,.5,.025])
fig.colorbar(im,cax=cbax,orientation='horizontal').set_label('Height above ground (m) | shared colour scale',fontsize=11)
fig.savefig(OUT/'different_urban_lidar_comparison.png',dpi=160,facecolor='white');plt.close(fig)
print(json.dumps({'selected':best,'raised':audit['raised_surface_diagnostic']},indent=2))
