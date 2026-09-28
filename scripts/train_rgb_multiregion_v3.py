"""Controlled multi-region classification continuation; production height is read-only."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import msvcrt
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml
import train_rgb_segmenter as legacy
from evaluate_rgb_segmenter_checkpoint import (
    KNOWN_EXPERIMENT, load_known_checkpoint, KNOWN_CHECKPOINT_SHA256,
    MetricGroup, independent_confusion, verify_file_identities, capture_file_identities,
)
from train_rgb_sampling_v2 import write_checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from msr.data.gamus_rgb_segmentation import GamusRgbSegmentationDataset, source_metadata_snapshot, canonical_sha256, authenticate_validation_content
from msr.data.rgb_multiregion_v3 import OemClassificationDataset, MixedClassificationDataset, epoch_indices
from msr.models.rgb_segmenter import RgbSegformer
from msr.evaluation.rgb_segmentation import evaluate_rgb_segmentation
from msr.evaluation.rgb_multiregion_v3 import compare

ACTIVE = ROOT / "outputs/orchestration/rgb_multiregion_v3_active.json"
EXTRA_SOURCES = ["scripts/train_rgb_multiregion_v3.py", "scripts/prepare_rgb_multiregion_v3.py",
    "scripts/train_rgb_sampling_v2.py", "scripts/evaluate_rgb_segmenter_checkpoint.py", "scripts/report_rgb_multiregion_v3.py",
    "src/msr/data/rgb_multiregion_v3.py", "src/msr/evaluation/rgb_multiregion_v3.py"]
REPLAY = ROOT / "outputs/evaluation/gamus_rgb_segmenter_v1_independent_replay/report.json"
REPLAY_SHA = "b471f67f5948a6c62299059054043c0ea93ad8ba8f8ffd9c3582d51fb49f06c3"

def acquire_lock():
    path=ROOT/"outputs/orchestration/rgb_multiregion_v3.lock"
    path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open("a+b")
    if stream.tell()==0:
        stream.write(b"0"); stream.flush()
    stream.seek(0)
    try: msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        stream.close()
        raise RuntimeError("Another multi-region V3 trainer holds the run lock")
    return stream

def sources():
    return {**legacy.source_manifest(), **{p:legacy.sha256(ROOT/p) for p in EXTRA_SOURCES}}

def gamus_data(base):
    d=base["data"]
    kw=dict(approved_index_path=legacy.resolve(d["approved_index_path"]),approved_index_sha256=d["approved_index_file_sha256"],rgb_scale=255.)
    return (GamusRgbSegmentationDataset(d["root"],"train",patch_size=512,random_crop=True,**kw),
            GamusRgbSegmentationDataset(d["root"],"val",patch_size=1024,**kw))

def new_model(payload):
    model=RgbSegformer.from_architecture(payload["architecture"])
    model.load_state_dict(payload["model_state_dict"],strict=True)
    return model.cuda()

def loader(dataset, workers, seed, **kw):
    return DataLoader(dataset,num_workers=workers,pin_memory=True,worker_init_fn=legacy.seed_worker,
        generator=torch.Generator().manual_seed(seed),**kw)

@torch.inference_mode()
def evaluate_oem(model,dataset,experiment,epoch):
    model.eval()
    overall,groups,identities,observed=MetricGroup(),{},[],[]
    for count,batch in enumerate(loader(dataset,2,19,batch_size=1),1):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits=model(batch["image"].cuda(non_blocking=True))["logits"]
        if logits.shape != (1,6,*batch["labels"].shape[-2:]) or not torch.isfinite(logits).all():
            raise RuntimeError("Invalid native OEM prediction")
        prediction=logits.argmax(1)[0].cpu().numpy()
        target=batch["labels"][0].numpy()
        valid=(batch["image_valid_mask"] & batch["classification_valid_mask"])[0].numpy()
        dark=batch["dark_pixel_proxy_mask"][0].numpy()
        matrix=independent_confusion(prediction,target,valid)
        overall.update(matrix,prediction,target,dark,valid)
        city=batch["city"][0]
        groups.setdefault(city,MetricGroup()).update(matrix,prediction,target,dark,valid)
        sample_id=batch["sample_id"][0]
        observed.append(sample_id)
        identities.append({"sample_id":sample_id,"valid":hashlib.sha256(valid.astype(np.uint8).tobytes()).hexdigest(),
            "labels":hashlib.sha256(np.where(valid,target,255).astype(np.uint8).tobytes()).hexdigest()})
        if count%10==0 or count==len(dataset):
            legacy.status(experiment,"oem_validation",epoch=epoch,completed=count,total=len(dataset))
    if observed != list(dataset.sample_ids):
        raise RuntimeError("OEM validation identities/order mismatch")
    def descriptions(value):
        if isinstance(value,dict): return {k:descriptions(v) for k,v in value.items()}
        return value.replace("GAMUS","OpenEarthMap") if isinstance(value,str) else value
    return descriptions({"overall":overall.compute(),"by_city":{c:g.compute() for c,g in groups.items()},
        "evaluated_sample_count":len(observed),"evaluated_ordered_ids_sha256":canonical_sha256(observed),
        "reference_grid_binding_sha256":canonical_sha256(identities),"interpretation":"New-region development validation; reused for checkpoint selection, not final test",
        "model_inputs":["RGB"],"height_evaluated":False})

def region_score(evaluation):
    return float(np.mean([g["six_class_identification"]["macro_f1"] for g in evaluation["by_city"].values()]))

def verify_binding(binding,config_path,experiment,gamus_train,gamus_val):
    if legacy.sha256(config_path)!=binding["config_sha256"] or legacy.sha256(experiment/"config.yaml")!=binding["config_sha256"] or sources()!=binding["sources"]:
        raise RuntimeError("Training code/config changed after sealing")
    for p,s in binding["sources"].items():
        if legacy.sha256(experiment/"source_snapshot"/p)!=s: raise RuntimeError("Source snapshot changed")
    for row in binding["files"]:
        if legacy.sha256(row["path"])!=row["sha256"]: raise RuntimeError(f"Protected input changed: {row['path']}")
    for row in binding["oem_metadata"]:
        st=Path(row["path"]).stat()
        if (st.st_size,st.st_mtime_ns)!=(row["size"],row["mtime_ns"]): raise RuntimeError("OEM source changed")
    if binding["gamus_metadata"] != {"train":source_metadata_snapshot(gamus_train),"val":source_metadata_snapshot(gamus_val)}:
        raise RuntimeError("GAMUS source changed")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/rgb_multiregion_v3.yaml")
    parser.add_argument("--resume",type=Path)
    args=parser.parse_args()
    config_path=ROOT/args.config
    config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_path=ROOT/config["base_config"]
    base=yaml.safe_load(base_path.read_text(encoding="utf-8"))
    settings=config["training"]
    if config["evaluation"]["app_promotion"] is not False or config["evaluation"]["height_changes"] is not False:
        raise RuntimeError("This experiment cannot change the app or height pipeline")
    if config["initial_checkpoint_sha256"] != KNOWN_CHECKPOINT_SHA256 or legacy.sha256(REPLAY)!=REPLAY_SHA:
        raise RuntimeError("Fixed baseline identity mismatch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bf16 is required")
    torch.set_num_threads(4)
    legacy.seed_everything(settings["seed"])
    legacy.verify_protected(base)
    initial=load_known_checkpoint(KNOWN_EXPERIMENT)
    destination=Path(config["data"]["destination"])
    manifest_path=destination/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    contract_path=destination/"contract.json"
    contract=json.loads(contract_path.read_text())
    if manifest["contract_sha256"]!=legacy.sha256(contract_path) or contract["config_sha256"]!=legacy.sha256(config_path):
        raise RuntimeError("Prepared data/config binding mismatch")
    gamus_train,gamus_val=gamus_data(base)
    if len(gamus_train)!=settings["gamus_draws"]:
        raise RuntimeError("Every GAMUS training tile must be replayed once per epoch")
    oem_train,oem_val=OemClassificationDataset(manifest_path,"train"),OemClassificationDataset(manifest_path,"val")
    train_hashes={r["image_sha256"] for r in oem_train.rows}
    if train_hashes & {r["image_sha256"] for r in oem_val.rows}:
        raise RuntimeError("Identical RGB content crosses training and validation")
    mixed=MixedClassificationDataset(gamus_train,oem_train,settings)
    oem_metadata=[]
    for row in manifest["pairs"]:
        for role in ("image","label"):
            path=Path(row[f"{role}_path"])
            if not path.resolve().is_relative_to(destination.resolve()) or legacy.sha256(path)!=row[f"{role}_sha256"]:
                raise RuntimeError("OEM file identity/path mismatch")
            st=path.stat()
            oem_metadata.append({"path":str(path),"size":st.st_size,"mtime_ns":st.st_mtime_ns})
    authenticate_validation_content(gamus_val,legacy.resolve(base["protocol"]["fixed_v3_independent_replay"]),
        base["protocol"]["fixed_v3_independent_replay_sha256"],cache_path=legacy.resolve(base["proof"]["validation_content_cache"]))
    files=[config_path,base_path,manifest_path,contract_path,REPLAY,
        KNOWN_EXPERIMENT/"checkpoint_best_guarded.pt",legacy.resolve(base["protocol"]["protected_checkpoint"]),legacy.resolve(base["protocol"]["live_pointer_file"])]
    binding={"config_sha256":legacy.sha256(config_path),"sources":sources(),
        "files":[{"path":str(p),"sha256":legacy.sha256(p)} for p in files],"oem_metadata":oem_metadata,
        "gamus_metadata":{"train":source_metadata_snapshot(gamus_train),"val":source_metadata_snapshot(gamus_val)},
        "software":legacy.software_versions(),"automatic_promotion":False}
    experiment=args.resume.resolve() if args.resume else ROOT/"experiments"/(legacy.run_id()+"_rgb_multiregion_v3")
    if args.resume:
        if json.loads((experiment/"binding.json").read_text())!=binding: raise RuntimeError("Resume binding mismatch")
    else:
        experiment.mkdir(parents=True,exist_ok=False)
        shutil.copy2(config_path,experiment/"config.yaml")
        for name in binding["sources"]:
            target=experiment/"source_snapshot"/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
        legacy.atomic_json(experiment/"binding.json",binding)
    legacy.atomic_json(ACTIVE,{"experiment":str(experiment),"config":str(config_path),"started_utc":legacy.utc_now(),"promotion_eligible":False})
    verify_binding(binding,config_path,experiment,gamus_train,gamus_val)
    print(f"Experiment: {experiment}\nTraining: 3837 GAMUS + 134 OEM tiles; validation: 859 GAMUS + 50 OEM. Heights protected.",flush=True)
    try:
        baseline_path=experiment/"baseline.json"
        if not baseline_path.exists():
            legacy.status(experiment,"baseline_validation")
            model=new_model(initial).eval()
            baseline={"gamus":json.loads(REPLAY.read_text())["evaluation"],"oem":evaluate_oem(model,oem_val,experiment,0)}
            legacy.atomic_json(baseline_path,baseline)
            legacy.atomic_json(experiment/"baseline.sha256.json",{"sha256":legacy.sha256(baseline_path)})
            del model
            torch.cuda.empty_cache()
        if legacy.sha256(baseline_path)!=json.loads((experiment/"baseline.sha256.json").read_text())["sha256"]:
            raise RuntimeError("Stored baseline changed")
        baseline=json.loads(baseline_path.read_text())
        compare(baseline,baseline,config["evaluation"])
        criterion=legacy.SixClassFocalLoss(settings["class_weights"],settings["focal_gamma"]).cuda()
        proof_path=experiment/"proof.json"
        if not proof_path.exists():
            legacy.status(experiment,"training_only_gradient_proof")
            model=new_model(initial)
            optimizer=torch.optim.AdamW(model.parameter_groups(settings["encoder_learning_rate"],settings["decoder_learning_rate"],settings["weight_decay"]))
            batch=next(iter(loader(Subset(mixed,[0,1,len(gamus_train),len(gamus_train)+1]),0,19,batch_size=4)))
            model.eval()
            with torch.no_grad(): before=float(legacy.loss_for(model,batch,criterion,torch.device("cuda"),"bf16"))
            maximum_gradient=0.
            for step in range(config["proof"]["steps"]):
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss=legacy.loss_for(model,batch,criterion,torch.device("cuda"),"bf16")
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite proof loss")
                loss.backward()
                gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                maximum_gradient=max(maximum_gradient,gradient)
                optimizer.step()
            model.eval()
            with torch.no_grad(): after=float(legacy.loss_for(model,batch,criterion,torch.device("cuda"),"bf16"))
            reduction=(before-after)/max(before,1e-12)
            proof={"passes":math.isfinite(after) and reduction>=config["proof"]["minimum_loss_reduction_fraction"] and maximum_gradient>0,
                "initial_loss":before,"final_loss":after,"loss_reduction":reduction,"gradient_norm":maximum_gradient,
                "proof_weights_discarded":True,"sample_ids":batch["sample_id"],"binding_sha256":canonical_sha256(binding)}
            legacy.atomic_json(proof_path,proof)
            del model,optimizer,batch
            torch.cuda.empty_cache()
        proof=json.loads(proof_path.read_text())
        if not proof["passes"] or proof["binding_sha256"]!=canonical_sha256(binding): raise RuntimeError("Training-only proof failed")
        legacy.seed_everything(settings["seed"])
        model=new_model(initial)
        del initial
        optimizer=torch.optim.AdamW(model.parameter_groups(settings["encoder_learning_rate"],settings["decoder_learning_rate"],settings["weight_decay"]))
        steps=settings["epochs"]*math.ceil((settings["gamus_draws"]+settings["oem_draws"])/settings["batch_size"])
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:legacy.learning_rate_factor(step,steps,settings["warmup_optimizer_steps"],settings["minimum_learning_rate_factor"]))
        history,best_safe,best_development=[],None,None
        commit_path=experiment/"commit.json"
        if commit_path.exists():
            commit=json.loads(commit_path.read_text())
            name=f"checkpoint_epoch_{commit['epoch']:03d}.pt"
            path=experiment/name
            if name!=commit["checkpoint"] or legacy.sha256(path)!=commit["sha256"]: raise RuntimeError("Checkpoint commit mismatch")
            saved=torch.load(path,map_location="cpu",weights_only=False)
            if saved["binding"]!=binding or saved["epoch"]!=commit["epoch"] or saved["promotion_eligible"] is not False: raise RuntimeError("Checkpoint binding mismatch")
            model.load_state_dict(saved["model_state_dict"],strict=True)
            optimizer.load_state_dict(saved["optimizer_state_dict"]); scheduler.load_state_dict(saved["scheduler_state_dict"])
            legacy.restore_rng(saved["rng_state"],torch.Generator())
            history,best_safe,best_development=saved["history"],saved["best_safe"],saved["best_development"]
            if [r["epoch"] for r in history]!=list(range(1,commit["epoch"]+1)): raise RuntimeError("Invalid checkpoint history")
            del saved
            shutil.copy2(path,experiment/"checkpoint_latest.pt")
            legacy.atomic_json(experiment/"checkpoint_latest.sha256.json",{"sha256":commit["sha256"]})
            legacy.write_history(experiment/"metrics.jsonl",history)
            # Repair aliases if a prior interruption happened after the commit.
            for alias,best in (("best_safe",best_safe),("best_development",best_development)):
                if best is not None:
                    best_path=experiment/f"checkpoint_epoch_{best['epoch']:03d}.pt"
                    expected=json.loads(best_path.with_suffix(".sha256.json").read_text())["sha256"]
                    if legacy.sha256(best_path)!=expected: raise RuntimeError("Best checkpoint integrity failure")
                    shutil.copy2(best_path,experiment/f"checkpoint_{alias}.pt")
                    legacy.atomic_json(experiment/f"checkpoint_{alias}.sha256.json",{"sha256":expected})
        for epoch in range(len(history)+1,settings["epochs"]+1):
            if history and len(history)>=settings["early_stopping_min_epochs"] and len(history)-best_development["epoch"]>=settings["early_stopping_patience"]:
                break
            verify_binding(binding,config_path,experiment,gamus_train,gamus_val)
            indices=epoch_indices(len(gamus_train),oem_train.rows,settings["oem_draws"],settings["seed"],epoch)
            train_loader=loader(mixed,settings["num_workers"],settings["seed"]+epoch*2003,sampler=indices,batch_size=settings["batch_size"])
            model.train(); losses=[]; started=time.monotonic()
            for i,batch in enumerate(train_loader,1):
                optimizer.zero_grad(set_to_none=True)
                loss=legacy.loss_for(model,batch,criterion,torch.device("cuda"),"bf16")
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),settings["max_gradient_norm"],error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
                if i==1 or i%20==0 or i==len(train_loader):
                    legacy.status(experiment,"training",epoch=epoch,total_epochs=settings["epochs"],batch=i,total_batches=len(train_loader),loss=losses[-1])
                if i==1 or i%200==0: print(f"Epoch {epoch}/{settings['epochs']} | {i}/{len(train_loader)} | loss {losses[-1]:.4f}",flush=True)
            def progress(update):
                if update["completed_batches"]%20==0:
                    legacy.status(experiment,"gamus_validation",epoch=epoch,**update)
            gamus_evaluation=evaluate_rgb_segmentation(model,loader(gamus_val,2,19,batch_size=1),"cuda",expected_ids=list(gamus_val.sample_ids),precision="bf16",native_size=1024,progress_callback=progress)
            evaluation={"gamus":gamus_evaluation,"oem":evaluate_oem(model,oem_val,experiment,epoch)}
            gate=compare(evaluation,baseline,config["evaluation"])
            score=region_score(evaluation["oem"])
            new_dev=best_development is None or score>best_development["score"]
            new_safe=gate["safety_pass"] and (best_safe is None or score>best_safe["score"])
            if new_dev: best_development={"epoch":epoch,"score":score}
            if new_safe: best_safe={"epoch":epoch,"score":score,"benefit_pass":gate["benefit_pass"]}
            row={"epoch":epoch,"validation":evaluation,"gate":gate,"train_loss":float(np.mean(losses)),
                 "elapsed_seconds":time.monotonic()-started,"sample_order_sha256":canonical_sha256(indices),"updated_utc":legacy.utc_now(),"promotion_eligible":False}
            history.append(row)
            verify_binding(binding,config_path,experiment,gamus_train,gamus_val)
            payload={"model_type":model.model_type,"architecture":model.architecture,"model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"rng_state":legacy.rng_state(torch.Generator()),
                "binding":binding,"epoch":epoch,"history":history,"best_safe":best_safe,"best_development":best_development,
                "input_contract":"raw_rgb_01_imagenet_normalized_in_model","promotion_eligible":False}
            name=f"checkpoint_epoch_{epoch:03d}.pt"
            write_checkpoint(experiment,name,payload)
            legacy.atomic_json(commit_path,{"epoch":epoch,"checkpoint":name,"sha256":legacy.sha256(experiment/name)})
            for alias,needed in (("latest",True),("best_development",new_dev),("best_safe",new_safe)):
                if needed:
                    shutil.copy2(experiment/name,experiment/f"checkpoint_{alias}.pt")
                    legacy.atomic_json(experiment/f"checkpoint_{alias}.sha256.json",{"sha256":legacy.sha256(experiment/name)})
            legacy.write_history(experiment/"metrics.jsonl",history)
            legacy.status(experiment,"epoch_complete",epoch=epoch,oem_region_macro_f1=score,safety_pass=gate["safety_pass"],benefit_pass=gate["benefit_pass"])
            print(f"Epoch {epoch} | OEM region F1 {score:.2%} | GAMUS F1 {gamus_evaluation['overall']['six_class_identification']['macro_f1']:.2%} | safety {gate['safety_pass']} | benefit {gate['benefit_pass']}",flush=True)
            if epoch>=settings["early_stopping_min_epochs"] and epoch-best_development["epoch"]>=settings["early_stopping_patience"]: break
        legacy.verify_protected(base)
        legacy.atomic_json(experiment/"outcome.json",{"completed_epochs":len(history),"best_development":best_development,"best_safe":best_safe,
            "app_promotion":False,"height_pipeline_unchanged":True,"final_holdout_consumed":False,
            "next":"Review per-class safety and visual errors; freeze an accepted candidate before fresh final-geography testing."})
        from report_rgb_multiregion_v3 import report
        report(experiment)
        legacy.status(experiment,"complete",epochs=len(history),best_safe=best_safe,best_development=best_development)
    except Exception as exc:
        legacy.status(experiment,"failed",error=repr(exc),resume_command=f".venv\\Scripts\\python.exe scripts/train_rgb_multiregion_v3.py --resume {experiment}")
        raise

if __name__=="__main__":
    with acquire_lock():
        main()
