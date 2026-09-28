"""Versioned RGB-only mixed-domain data; independent masks survive augmentation."""
from __future__ import annotations
import json
from pathlib import Path
import warnings
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CROSSWALK = np.array([255, 0, 4, 0, 3, 5, 2, 255, 1], np.uint8)

def augment_sample(sample, size=384, crop_min=256, crop_max=512, photometric=True):
    """One aligned spatial transform; colour never changes validity or labels."""
    out = dict(sample)
    height, width = sample["labels"].shape
    maximum = min(height, width, crop_max)
    minimum = min(crop_min, maximum)
    side = int(torch.randint(minimum, maximum+1, ()).item())
    top = int(torch.randint(height-side+1, ()).item())
    left = int(torch.randint(width-side+1, ()).item())
    turns, flip = int(torch.randint(4, ()).item()), bool(torch.randint(2, ()).item())
    for key in ("image", "labels", "image_valid_mask", "classification_valid_mask", "dark_pixel_proxy_mask"):
        value = sample[key][..., top:top+side, left:left+side]
        is_image = key == "image"
        original_dtype = value.dtype
        value = value.unsqueeze(0) if is_image else value[None,None]
        if is_image:
            value = F.interpolate(value.float(), (size,size), mode="bilinear", align_corners=False, antialias=True)[0]
        else:
            value = F.interpolate(value.float(), (size,size), mode="nearest")[0,0].to(original_dtype)
        value = torch.rot90(value, turns, (-2,-1))
        if flip:
            value = value.flip(-1)
        out[key] = value.contiguous()
    image = out["image"]
    if photometric:
        # Modest sensor/illumination variation, not drastic recolouring.
        brightness = .85 + .3*torch.rand(())
        contrast = .85 + .3*torch.rand(())
        saturation = .8 + .4*torch.rand(())
        grey = image.mean(0, keepdim=True)
        image = ((grey + saturation*(image-grey) - .5)*contrast + .5)*brightness
        if torch.rand(()) < .2:
            image = F.avg_pool2d(image[None],3,stride=1,padding=1)[0]
    out["image"] = image.clamp(0,1).masked_fill(~out["image_valid_mask"][None],0)
    # Proxy remains attached to original RGB, never augmented colour.
    out["dark_pixel_proxy_mask"] &= out["image_valid_mask"]
    return out

class OemClassificationDataset(Dataset):
    def __init__(self, manifest_path, split):
        if split not in ("train","val"):
            raise ValueError("Only development train/val are supported")
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.rows = [r for r in payload["pairs"] if r["split"] == split]
        self.sample_ids = tuple(r["sample_id"] for r in self.rows)
        if not self.rows or len(set(self.sample_ids)) != len(self.rows):
            raise ValueError("Nonempty unique identities required")
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,index):
        row = self.rows[index]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",NotGeoreferencedWarning)
            with rasterio.open(row["image_path"]) as image, rasterio.open(row["label_path"]) as label:
                if image.count != 3 or image.dtypes != ("uint8",)*3 or image.shape != label.shape or image.crs != label.crs or not image.transform.almost_equals(label.transform):
                    raise ValueError("RGB/reference grid or radiometry mismatch")
                rgb = image.read().astype(np.float32)/255
                raw = label.read(1)
                if raw.dtype.kind not in "iu" or np.any(raw > 8):
                    raise ValueError("Invalid OEM class IDs")
                image_valid = np.all(image.read_masks()>0,axis=0) & np.isfinite(rgb).all(0)
                labels = CROSSWALK[raw].astype(np.int64)
                class_valid = (label.read_masks(1)>0) & (labels != 255)
        rgb[:,~image_valid] = 0
        return {"image":torch.from_numpy(rgb), "labels":torch.from_numpy(labels),
            "image_valid_mask":torch.from_numpy(image_valid), "classification_valid_mask":torch.from_numpy(class_valid),
            "dark_pixel_proxy_mask":torch.from_numpy(image_valid & (rgb.mean(0)<=.15)),
            "sample_id":row["sample_id"],"city":row["region"],"region":row["region"]}

class MixedClassificationDataset(Dataset):
    def __init__(self,gamus,oem,settings):
        self.gamus,self.oem,self.settings = gamus,oem,settings
    def __len__(self):
        return len(self.gamus)+len(self.oem)
    def __getitem__(self,index):
        sample = self.gamus[index] if index<len(self.gamus) else self.oem[index-len(self.gamus)]
        return augment_sample(sample,self.settings["crop_size"],self.settings["source_crop_min"],self.settings["source_crop_max"])

def epoch_indices(gamus_count,oem_rows,oem_draws,seed,epoch):
    """Every GAMUS training tile once; additional equal-city OEM draws."""
    generator = torch.Generator().manual_seed(seed+1009*epoch)
    cities = sorted({r["region"] for r in oem_rows})
    groups = [[i for i,r in enumerate(oem_rows) if r["region"]==city] for city in cities]
    indices = list(range(gamus_count))
    for _ in range(oem_draws):
        group = groups[int(torch.randint(len(groups),(),generator=generator))]
        indices.append(gamus_count+group[int(torch.randint(len(group),(),generator=generator))])
    order = torch.randperm(len(indices),generator=generator).tolist()
    return [indices[i] for i in order]
