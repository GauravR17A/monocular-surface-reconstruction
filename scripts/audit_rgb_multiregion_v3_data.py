"""Visual alignment/contact sheet and label balance; never model-based selection."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rasterio
from stage_oem_all_six_scenes import CROSSWALK, save_json
from evaluate_rgb_segmenter_checkpoint import CLASS_NAMES, COLORS

DATA=Path("D:/MSRData/training/oem_multiregion_v3")

def main():
    if (DATA/"manifest.json").exists():
        rows=json.loads((DATA/"manifest.json").read_text())["pairs"]
    else:
        plan=json.loads((DATA/"download_plan.json").read_text())
        rows=[]
        for row in plan["pairs"]:
            row=dict(row)
            for role in ("image","label"):
                row[f"{role}_path"]=str(DATA/row["split"]/row["region"]/(role+"s")/Path(row[f"{role}_member"]).name)
            if all(Path(row[f"{role}_path"]).exists() for role in ("image","label")): rows.append(row)
    regions=sorted({r["region"] for r in rows})
    chosen=[next(r for r in rows if r["region"]==region) for region in regions]
    colors=np.full((256,3),60,np.uint8)
    for i,color in enumerate(COLORS): colors[i]=[int(color[j:j+2],16) for j in (1,3,5)]
    figure,axes=plt.subplots(len(chosen),2,figsize=(10,4.8*len(chosen)),squeeze=False)
    for (left,right),row in zip(axes,chosen):
        with rasterio.open(row["image_path"]) as image, rasterio.open(row["label_path"]) as label:
            rgb=image.read().transpose(1,2,0)
            mapped=CROSSWALK[label.read(1)]
        left.imshow(rgb); right.imshow(colors[mapped])
        left.set_title(f"{row['sample_id']} | {row['split']} | RGB")
        right.set_title("Mapped reference; grey = ignored")
        left.axis("off"); right.axis("off")
    figure.legend(handles=[Patch(color=c,label=n.replace("_"," ")) for c,n in zip(COLORS,CLASS_NAMES)],loc="lower center",ncol=3)
    figure.tight_layout(rect=(0,.03,1,1))
    figure.savefig(DATA/"data_alignment_audit.jpg",dpi=90)
    plt.close(figure)
    if all("class_pixels" in r for r in rows):
        balance={s:dict(zip(CLASS_NAMES,np.sum([r["class_pixels"] for r in rows if r["split"]==s],axis=0).tolist())) for s in ("train","val")}
        save_json(DATA/"label_balance.json",{"class_pixels":balance,"all_six_supported":all(min(counts.values())>0 for counts in balance.values()),"preview_ids":[r["sample_id"] for r in chosen]})
    print(DATA/"data_alignment_audit.jpg")

if __name__=="__main__": main()
