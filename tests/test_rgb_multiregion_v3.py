"""Safety tests for the isolated multi-region classifier experiment."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"src"))
from msr.data.rgb_multiregion_v3 import augment_sample, epoch_indices, CROSSWALK, OemClassificationDataset
from msr.evaluation.rgb_multiregion_v3 import compare
from prepare_rgb_multiregion_v3 import validate_splits
from evaluate_rgb_segmenter_checkpoint import MetricGroup

CONFIG=yaml.safe_load((ROOT/"configs/rgb_multiregion_v3.yaml").read_text())

def sample():
    labels=torch.arange(64).reshape(8,8)%6
    image_valid=torch.ones(8,8,dtype=torch.bool); image_valid[:2]=False
    class_valid=torch.ones(8,8,dtype=torch.bool); class_valid[:,2:4]=False
    labels[~class_valid]=255
    return {"image":torch.rand(3,8,8),"labels":labels,"image_valid_mask":image_valid,
            "classification_valid_mask":class_valid,"dark_pixel_proxy_mask":torch.zeros(8,8,dtype=torch.bool),"sample_id":"fixture"}

def test_crosswalk_ignores_missing_and_agriculture():
    assert CROSSWALK.tolist()==[255,0,4,0,3,5,2,255,1]

def test_independent_masks_and_ids_survive_augmentation():
    x=sample()
    torch.manual_seed(1); a=augment_sample(x,16,8,8,photometric=False)
    torch.manual_seed(1); b=augment_sample(x,16,8,8,photometric=True)
    for key in ("labels","image_valid_mask","classification_valid_mask"):
        assert torch.equal(a[key],b[key])
    assert set(b["labels"].unique().tolist()) <= {0,1,2,3,4,5,255}
    assert torch.equal(b["labels"]==255,~b["classification_valid_mask"])
    assert not torch.equal(b["classification_valid_mask"],b["image_valid_mask"])
    assert torch.all(b["image"][:,~b["image_valid_mask"]]==0)
    assert b["image"].min()>=0 and b["image"].max()<=1

def test_every_gamus_tile_replayed_and_oem_indices_bounded():
    rows=[{"region":"a"},{"region":"a"},{"region":"b"}]
    indices=epoch_indices(12,rows,8,20,1)
    assert len(indices)==20 and sorted(i for i in indices if i<12)==list(range(12))
    assert all(0<=i<15 for i in indices)
    assert indices==epoch_indices(12,rows,8,20,1)
    assert indices!=epoch_indices(12,rows,8,20,2)

def test_split_regions_are_disjoint():
    validate_splits(CONFIG["data"])
    data=deepcopy(CONFIG["data"]); data["train_regions"]["melbourne"]=1
    with pytest.raises(ValueError): validate_splits(data)

def evidence():
    group=MetricGroup().compute()
    group["six_class_identification"]["macro_f1"]=.7
    for row in group["six_class_identification"]["per_class"].values():
        row.update(f1=.7,support_pixels=100)
    group["road_boundary_quality"]["f1"]=.4
    return {d:{"overall":deepcopy(group),"by_city":{"city":deepcopy(group)},
               "evaluated_ordered_ids_sha256":"ids","reference_grid_binding_sha256":"grid"} for d in ("gamus","oem")}

def test_no_improvement_is_not_success():
    a=evidence(); out=compare(a,a,CONFIG["evaluation"])
    assert out["safety_pass"] and not out["benefit_pass"] and not out["app_promotion"]

@pytest.mark.parametrize("category",["ground","buildings","water","roads","low_vegetation","trees"])
def test_every_gamus_class_protected_per_city(category):
    a=evidence(); b=deepcopy(a)
    b["gamus"]["by_city"]["city"]["six_class_identification"]["per_class"][category]["f1"]-=.02
    assert not compare(b,a,CONFIG["evaluation"])["safety_pass"]

def test_oem_benefit_does_not_override_gamus_regression():
    a=evidence(); b=deepcopy(a)
    for group in [b["oem"]["overall"],b["oem"]["by_city"]["city"]]:
        group["six_class_identification"]["macro_f1"]+=.1
        for row in group["six_class_identification"]["per_class"].values(): row["f1"]+=.1
    b["gamus"]["overall"]["six_class_identification"]["macro_f1"]-=.1
    out=compare(b,a,CONFIG["evaluation"])
    assert out["benefit_pass"] and not out["eligible_development_candidate"]

def test_reference_binding_cannot_change():
    a=evidence(); b=deepcopy(a); b["oem"]["reference_grid_binding_sha256"]="other"
    with pytest.raises(ValueError): compare(b,a,CONFIG["evaluation"])

def test_road_and_dark_water_safety():
    a=evidence(); b=deepcopy(a)
    b["oem"]["overall"]["road_boundary_quality"]["f1"]-=.05
    b["gamus"]["overall"]["water_dark_pixel_proxy"]["false_water_rate_on_dark_non_water"]+=.02
    out=compare(b,a,CONFIG["evaluation"])
    assert not out["safety_pass"] and len(out["failed_checks"])==2

def test_oem_dataset_uses_separate_image_and_label_validity(tmp_path):
    rgb=np.ones((3,32,32),np.uint8)*90
    rgb[:,:2,:]=0
    labels=np.ones((32,32),np.uint8)*8
    labels[:,8:10]=7
    for role,array in (("image",rgb),("label",labels[None])):
        with rasterio.open(tmp_path/f"{role}.tif","w",driver="GTiff",width=32,height=32,count=array.shape[0],dtype="uint8",nodata=0,transform=from_origin(0,32,1,1)) as dst: dst.write(array)
    payload={"pairs":[{"split":"train","sample_id":"fixture","region":"test","image_path":str(tmp_path/"image.tif"),"label_path":str(tmp_path/"label.tif")}]}
    manifest=tmp_path/"manifest.json"; manifest.write_text(json.dumps(payload))
    result=OemClassificationDataset(manifest,"train")[0]
    assert not result["image_valid_mask"][:2].any()
    assert result["classification_valid_mask"][:2,:8].all()
    assert not result["classification_valid_mask"][:,8:10].any()
    assert result["image_valid_mask"][2:,8:10].all()
    assert "height" not in result

def test_release_and_height_updates_forbidden():
    assert CONFIG["evaluation"]["app_promotion"] is False
    assert CONFIG["evaluation"]["height_changes"] is False
