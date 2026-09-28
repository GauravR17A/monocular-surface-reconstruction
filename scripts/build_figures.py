"""Rebuild explanatory architecture and recorded research plots (no inference)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'docs/assets';ASSETS.mkdir(exist_ok=True)
INK='#18313d';TEAL='#087f83';BLUE='#447ba9';MUTED='#526975'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED,'savefig.facecolor':'white'})

fig,ax=plt.subplots(figsize=(11,6.5));ax.set_xlim(0,11);ax.set_ylim(0,6.5);ax.axis('off')
ax.text(.2,6.14,'From one image to inspectable surface products',fontsize=19,weight='bold')
ax.text(.2,5.76,'Model outputs, calibration, classification and display remain distinct.',fontsize=11,color=MUTED)
def box(x,y,w,h,title,body,color=TEAL):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10,rounding_size=0.06',fc='#f1f7f8',ec=color,lw=1.2))
 ax.text(x+.12,y+h-.27,title,fontsize=10.5,weight='bold',color=color)
 ax.text(x+.12,y+h-.56,body,fontsize=8.7,va='top',linespacing=1.45)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.2,color=MUTED))
box(.3,3.2,2.0,1.42,'RGB input','Decode / valid mask\nIntensity normalisation\nSpatial grid, if supplied')
box(3,4.1,2.55,1.0,'Relative geometry','Depth Anything V2 Small\nNo automatic metre scale')
box(3,2.65,2.55,1.02,'Guarded height model','Protected urban backbone\nVegetation-aware fusion')
box(3,.94,2.55,1.03,'Independent RGB labels','Six-category classifier\nNo height modification',BLUE)
box(6.22,3.07,2.1,1.22,'Scientific products','Raw height + probabilities\nRelative-depth raster\nIndependent valid masks')
box(6.22,1.11,2.1,1.17,'Optional calibration','DEM or control points\nExplicit absolute datum\nIndependent validation')
box(9.03,2.53,1.55,1.6,'3D workspace','Derived mesh\nInspection / profiles\nNavigation\nRaw export links')
arrow((2.4,4.2),(2.9,4.54));arrow((2.4,3.82),(2.9,3.2));arrow((2.3,3.2),(2.95,1.65))
arrow((4.28,4.0),(4.28,3.77));arrow((5.66,3.23),(6.12,3.66));arrow((7.27,2.98),(7.27,2.39));arrow((8.44,3.68),(8.93,3.52));arrow((8.44,1.67),(9.42,2.43));arrow((5.65,1.32),(8.94,2.67))
ax.text(.3,.3,'Reference data evaluates a frozen output. Display styling never rewrites the raw height product.',fontsize=9.5,color=MUTED)
fig.subplots_adjust(left=.02,right=.98,top=.99,bottom=.02)
fig.savefig(ASSETS/'architecture.png',dpi=190);fig.savefig(ASSETS/'architecture.svg');plt.close(fig)

data=json.loads((ROOT/'docs/evidence/research-metrics.json').read_text(encoding='utf-8'))
rows=data['height_candidates'];labels=[x['label'] for x in rows];x=list(range(len(rows)))
fig,axes=plt.subplots(1,2,figsize=(10.8,5.1),sharey=True)
for ax,key,title,color in zip(axes,['building_rmse_m','canopy_rmse_m'],['Measured building support','Vegetation support'],[TEAL,BLUE]):
 values=[r[key] for r in rows];ax.barh(x,values,color=[INK]+[color]*5,height=.57)
 ax.set_yticks(x,labels);ax.invert_yaxis();ax.set_title(title,loc='left',fontweight='bold',pad=13);ax.set_xlabel('RMSE (m) · lower is better');ax.set_xlim(0,max(values)*1.22)
 ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 for y,v in zip(x,values):ax.text(v+.12,y,f'{v:.2f}',va='center',fontsize=9)
fig.suptitle('Lower averages did not establish an eligible replacement',x=.03,ha='left',fontsize=16,fontweight='bold')
fig.text(.03,.025,'Development comparisons. Different selected experiments; their paired controls and safety guards still govern promotion.\nThe protected model remains the released height predictor. Source protocols are indexed in Results and Experiments.',fontsize=9,color=MUTED)
fig.subplots_adjust(left=.21,right=.98,top=.82,bottom=.2,wspace=.28);fig.savefig(ASSETS/'height-experiments.png',dpi=190);plt.close(fig)
print('Built architecture (PNG/SVG) and height comparison plot.')
