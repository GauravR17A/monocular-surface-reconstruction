import io
import tarfile

import numpy as np
from PIL import Image

from msr.data.webdataset_dataset import HeightSampleTransform, make_webdataset


def _encoded_image(array, image_format):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=image_format)
    return buffer.getvalue()


def test_height_sample_transform_decodes_aligned_float_tiff():
    rgb = np.full((16, 16, 3), 127, dtype=np.uint8)
    height = np.zeros((16, 16), dtype=np.float32)
    height[4:12, 4:12] = 15.0
    sample = {
        "__key__": "Europe_France_Paris__tile_1",
        "jpg": _encoded_image(rgb, "JPEG"),
        "tiff": _encoded_image(height, "TIFF"),
    }

    output = HeightSampleTransform(
        patch_size=8,
        training=False,
        height_min_m=0.0,
        height_max_m=1000.0,
    )(sample)

    assert output["image"].shape == (3, 8, 8)
    assert output["height"].shape == (1, 8, 8)
    assert output["valid_mask"].all()
    assert output["building_mask"].all()
    assert output["region"] == "Europe_France_Paris"


def test_make_webdataset_groups_tar_members(tmp_path):
    rgb = _encoded_image(np.full((8, 8, 3), 80, dtype=np.uint8), "JPEG")
    height = _encoded_image(np.full((8, 8), 7.0, dtype=np.float32), "TIFF")
    shard = tmp_path / "sample.tar"
    with tarfile.open(shard, "w") as archive:
        for name, payload in (("Region__tile.jpg", rgb), ("Region__tile.tiff", height)):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    dataset = make_webdataset(
        [str(shard)],
        patch_size=8,
        training=False,
        height_min_m=0.0,
        height_max_m=1000.0,
        seed=1,
    )
    output = next(iter(dataset))

    assert output["sample_id"] == "Region__tile"
    assert output["height"].shape == (1, 8, 8)


def test_guided_tall_crop_contains_tall_pixel():
    rgb = np.full((32, 32, 3), 127, dtype=np.uint8)
    height = np.zeros((32, 32), dtype=np.float32)
    height[1, 1] = 45.0
    sample = {
        "__key__": "Europe_France_Paris__tall_tile",
        "jpg": _encoded_image(rgb, "JPEG"),
        "tiff": _encoded_image(height, "TIFF"),
    }

    output = HeightSampleTransform(
        patch_size=8,
        training=True,
        height_min_m=0.0,
        height_max_m=1000.0,
        foreground_crop_probability=1.0,
        tall_crop_probability=1.0,
        tall_threshold_m=20.0,
    )(sample)

    assert output["height"].max().item() == 45.0
    assert output["building_mask"].sum().item() == 1.0
