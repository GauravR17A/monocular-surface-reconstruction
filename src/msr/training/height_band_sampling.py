"""Train-only, deterministic coverage plus reference-height stratified sampling.

References choose training examples/crop anchors, never model inputs or inference
routes. Crops retain ALL valid labels; a requested height band is not a loss mask.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

BANDS = ('ground','short_building','tall_building','short_vegetation','medium_vegetation','tall_vegetation')
SOURCE_BANDS = {'urban':('short_building','tall_building'),
                'forest':('ground','short_vegetation','medium_vegetation','tall_vegetation')}


def height_masks(target, valid, domain):
    target=np.asarray(target); valid=np.asarray(valid,dtype=bool); domain=np.asarray(domain)
    if target.shape != valid.shape or domain.shape != target.shape:
        raise ValueError('Height/domain/support grids must match')
    if not np.isfinite(target[valid]).all():
        raise ValueError('Nonfinite valid reference')
    building=valid&(domain==1); vegetation=valid&(domain==2)
    return {'ground':valid&(domain==0), 'short_building':building&(target<20),
            'tall_building':building&(target>=20),
            'short_vegetation':vegetation&(target>0)&(target<=2),
            'medium_vegetation':vegetation&(target>2)&(target<15),
            'tall_vegetation':vegetation&(target>=15)}


def count_bands(target,valid,domain):
    return {name:int(mask.sum()) for name,mask in height_masks(target,valid,domain).items()}


def choose_crop(target,valid,domain,band,seed,patch=384):
    if target.ndim!=2 or min(target.shape)<patch:
        raise ValueError('Native image is smaller than training crop')
    mask=np.asarray(valid,dtype=bool) if band=='coverage' else height_masks(target,valid,domain)[band]
    points=np.argwhere(mask)
    if not len(points):
        raise ValueError('Requested crop has no height support')
    point=points[int(np.random.default_rng(seed).integers(len(points)))]
    row=max(0,min(int(point[0])-patch//2,target.shape[0]-patch))
    col=max(0,min(int(point[1])-patch//2,target.shape[1]-patch))
    return {'row':row,'col':col,'size':patch,'seed':int(seed),'band':band}


def coverage_draws(items,epoch,seed,count=300):
    # Across two epochs, every 600 forest / 450 urban image is seen at least once.
    order=np.random.default_rng(seed).permutation(len(items))
    return [items[int(order[((epoch-1)*count+i)%len(order)])] for i in range(count)]


def balanced_source_plan(items,source,epoch,seed,draws=600,minimum_pixels=64):
    if draws!=600 or source not in SOURCE_BANDS:
        raise ValueError('Only the declared 600-draw two-source plan is supported')
    rows=[{'sample_id':item['sample_id'],'band':'coverage'} for item in coverage_draws(items,epoch,seed)]
    rng=np.random.default_rng(seed+epoch*101)
    bands=SOURCE_BANDS[source]
    for band in bands:
        pool=[item for item in items if item['counts'][band]>=minimum_pixels]
        if not pool:
            raise ValueError(f'No eligible training support for {source}/{band}')
        weights=np.array([item['counts'][band]/max(item['supported_pixels'],1) for item in pool],dtype=np.float64)
        weights/=weights.sum()
        for index in rng.choice(len(pool),300//len(bands),replace=True,p=weights):
            rows.append({'sample_id':pool[int(index)]['sample_id'],'band':band})
    rng.shuffle(rows)
    for index,row in enumerate(rows):
        row['seed']=int(seed+epoch*10000+index*2)
    return rows


def summarize_draws(draws,index):
    counts=Counter(row['band'] for row in draws)
    return {'draws':len(draws),'unique_images':len({r['sample_id'] for r in draws}),
            'requested_bands':dict(counts),
            'regions':dict(Counter(index[row['sample_id']]['region'] for row in draws))}
