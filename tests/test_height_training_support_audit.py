import importlib.util
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location('height_support_audit',Path(__file__).resolve().parents[1]/'scripts/audit_height_training_support.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_bands_ignore_missing_and_do_not_clip_outliers():
    result=audit.height_bands([float('nan'),-1,0,2,5,10,20,40,80,200,201])
    assert list(result.values()) == [1]*8


def test_model_directory_is_not_a_provenance_binding():
    result=audit.prior_status(np.ones((2,2)),{'MSR_MODEL':'dav2'},(2,2))
    assert result['stored_01']
    assert not result['source_and_weights_hash_tags_present']
    assert not audit.prior_status(np.full((2,2),np.nan),{},(2,2))['stored_01']
    assert not audit.prior_status(np.ones((2,2)),{},(3,3))['shape_matches']


def test_annotation_support_is_local_and_does_not_turn_missing_to_zero():
    ann={'segmentation':[[1,1,3,1,3,3,1,3]]}
    target=np.full((5,5),9.,dtype=np.float32)
    valid=np.ones((5,5),dtype=bool); valid[1,1]=False
    values=audit.annotation_values(ann,target,valid)
    assert values.tolist() == [9,9,9]
    assert audit.annotation_values(ann,target,np.zeros((5,5),dtype=bool)).size == 0
