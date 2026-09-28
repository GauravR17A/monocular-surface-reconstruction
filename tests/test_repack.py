import importlib.util
import io
import tarfile
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "data" / "repack_highbuild_full.py"
SPEC = importlib.util.spec_from_file_location("repack_highbuild_full", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_repack_source_selects_and_atomically_writes_members(tmp_path):
    source_relative = "data/webdataset/train/source.tar"
    source = tmp_path / source_relative
    source.parent.mkdir(parents=True)
    with tarfile.open(source, "w") as archive:
        for name in ("wanted.jpg", "wanted.tiff", "wanted.json", "ignored.jpg"):
            payload = name.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    rows = {
        "train": [
            {
                "msr_shard": "shards/train/source-train.tar",
                "webdataset_image_member": "wanted.jpg",
                "webdataset_mask_member": "wanted.tiff",
                "webdataset_json_member": "wanted.json",
            }
        ]
    }
    counts = MODULE.repack_source(tmp_path, source_relative, rows)

    output = tmp_path / "msr_splits/shards/train/source-train.tar"
    assert counts == {"train": 1}
    assert output.is_file()
    assert not output.with_suffix(".tar.part").exists()
    with tarfile.open(output) as archive:
        assert set(archive.getnames()) == {"wanted.jpg", "wanted.tiff", "wanted.json"}
