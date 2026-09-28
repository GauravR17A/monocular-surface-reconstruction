"""CPU-only TRAIN exposure audit and sealed four-epoch sampling plans."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import warnings

import torch
from rasterio.errors import NotGeoreferencedWarning
from msr.training.height_band_sampling import count_bands,choose_crop,balanced_source_plan,summarize_draws

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('band_dev',ROOT/'scripts/train_class_assisted_height.py')
dev=importlib.util.module_from_spec(spec); spec.loader.exec_module(dev)
common=dev.common
CACHE=Path('D:/MSRData/experiments/class_assisted_height_development_v1/20260919T162009394247Z')


def arrays(record):
    sample=common.full_sample(record)
    return sample['height'][0].numpy(),sample['regression_mask'][0].numpy(),sample['domain_target'].numpy()


def execute(output):
    if output.exists():
        raise FileExistsError('Refusing to overwrite a sampling audit')
    proof=json.loads((ROOT/'configs/class_assisted_height_proof_v1.json').read_text())
    original_code=common.authenticate(proof)
    records=common.load_surface_manifest(common.resolve(proof['train_manifest']))
    receipts=json.loads((CACHE/'data_binding.json').read_text())
    if len(records)!=1050 or receipts['counts']['train']!=1050:
        raise ValueError('Pinned training population changed')
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    def scan(record):
        receipt=dev.receipts_file(CACHE,'train',record.sample_id)
        if common.sha(receipt)!=receipts['receipts']['train'][record.sample_id]:
            raise ValueError('Training receipt changed')
        dev.verify_receipt(receipt,record,files=True)
        target,valid,domain=arrays(record)
        return {'sample_id':record.sample_id,'landscape':record.landscape,'region':record.region,
                'shape':list(target.shape),'supported_pixels':int(valid.sum()),
                'counts':count_bands(target,valid,domain),'receipt_sha256':common.sha(receipt)}
    index={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item in pool.map(scan,records):
            index[item['sample_id']]=item
            if len(index)%100==0 or len(index)==len(records):
                print(f'Train coverage audited: {len(index)}/{len(records)}',flush=True)
    plans={arm:{} for arm in ('control','balanced')}
    grouped=defaultdict(list)
    for epoch in range(1,5):
        original=dev.epoch_plan(records,epoch,proof['seed'])
        for urban,forest,seed in original:
            pair=[]
            for offset,record in enumerate((urban,forest)):
                row={'sample_id':record.sample_id,'band':'coverage','seed':seed+offset}
                pair.append(row)
                grouped[record.sample_id].append(row)
            plans['control'].setdefault(str(epoch),[]).append(pair)
        sources={}
        for source,offset in (('urban',0),('forest',1000000)):
            items=[index[r.sample_id] for r in records if r.landscape==source]
            sources[source]=balanced_source_plan(items,source,epoch,proof['seed']+offset)
        for urban,forest in zip(sources['urban'],sources['forest']):
            plans['balanced'].setdefault(str(epoch),[]).append([urban,forest])
            grouped[urban['sample_id']].append(urban)
            grouped[forest['sample_id']].append(forest)
    def resolve_crops(record):
        target,valid,domain=arrays(record)
        for row in grouped[record.sample_id]:
            crop=choose_crop(target,valid,domain,row['band'],row['seed'])
            r,c,size=crop['row'],crop['col'],crop['size']
            cut=(slice(r,r+size),slice(c,c+size))
            row['crop']=crop
            row['exposure']=count_bands(target[cut],valid[cut],domain[cut])
            row['supported_pixels']=int(valid[cut].sum())
        return record.sample_id
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number,_ in enumerate(pool.map(resolve_crops,records),1):
            if number%100==0 or number==len(records):
                print(f'Actual crop exposure measured: {number}/{len(records)}',flush=True)
    exposure={}
    for arm,epochs in plans.items():
        exposure[arm]={}
        for epoch,pairs in epochs.items():
            exposure[arm][epoch]={}
            for source,position in (('urban',0),('forest',1)):
                draws=[pair[position] for pair in pairs]
                pixels=Counter()
                for row in draws:
                    pixels.update(row['exposure'])
                exposure[arm][epoch][source]={**summarize_draws(draws,index),'labelled_pixel_exposure':dict(pixels)}
        for until in (2,4):
            seen={row['sample_id'] for epoch,pairs in epochs.items() if int(epoch)<=until for pair in pairs for row in pair}
            if set(index)-seen:
                raise ValueError(f'{arm} lost complete training coverage by epoch {until}')
    if common.authenticate(proof)!=original_code:
        raise ValueError('Sources changed during preparation')
    # Source files read for sampling are exclusively from the pinned train manifest.
    report={'schema':'msr.height_band_exposure.v1','training_only':True,'validation_read':False,
            'final_tests_used':False,'training_manifest_sha256':proof['train_manifest_sha256'],
            'epochs':4,'pairs_per_epoch':600,'patch_size':384,'loss':'original_source_balanced',
            'conditioning':'uniform in BOTH arms','complete_train_coverage_by_epochs':[2,4],
            'seed':proof['seed'],'cache_data_binding_sha256':common.sha(CACHE/'data_binding.json'),
            'source_sha256':{**original_code,'scripts/prepare_height_band_sampling.py':common.sha(Path(__file__)),
                'src/msr/training/height_band_sampling.py':common.sha(ROOT/'src/msr/training/height_band_sampling.py')},
            'index':index,'exposure':exposure,
            'limitations':['Sampling uses reference heights only to select training observations, never forward inputs.',
                           'Forest chips remain native 384 pixels; balancing changes chip frequency, not their geometry.',
                           'Repeated crops are observations, not new independent images; regions are reported explicitly.']}
    common.atomic_json(output/'plans.json',plans)
    report['plans_sha256']=common.sha(output/'plans.json')
    common.atomic_json(output/'audit.json',report)
    print(json.dumps({'images':len(index),'plans_sha256':report['plans_sha256'],'epoch1_exposure':{a:exposure[a]['1'] for a in exposure}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('D:/MSRData/experiments/height_band_sampling_preparation_v1'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',NotGeoreferencedWarning)
        execute(parser.parse_args().output)
