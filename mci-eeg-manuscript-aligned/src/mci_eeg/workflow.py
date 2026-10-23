"""Identical observed/null statistics, immutable protocol and resumable fold logs."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import traceback
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from .data import load_dataset, fingerprint, file_hash
from .models import MODELS, outer_fold


def atomic_json(path,obj):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    for attempt in range(6):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt==5:raise
            time.sleep(0.02*2**attempt)


def seed_for(base,index):
    return int(np.random.SeedSequence([base,index]).generate_state(1)[0])


def splits(y,cfg):
    if np.bincount(y,minlength=2).min()<cfg["folds"]:
        raise ValueError("Not enough subjects per class for outer CV")
    for repeat in range(cfg["repeats"]):
        cv=StratifiedKFold(cfg["folds"],shuffle=True,random_state=cfg["seed"]+97*repeat)
        for fold,(train,test) in enumerate(cv.split(np.zeros(len(y)),y)):
            yield repeat,fold,train,test


def full_statistic(data,y,cfg,directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/"labels.json",dict(zip(data.ids.tolist(),y.tolist())))
    probabilities={p:np.full((len(y),cfg["repeats"]),np.nan) for p in MODELS}
    empty_count=0
    for repeat,fold,train,test in splits(y,cfg):
        path=directory/f"fold_{repeat:02d}_{fold:02d}.json"
        if path.exists():
            record=json.loads(path.read_text(encoding="utf-8"))
            if record["train"]!=train.tolist() or record["test"]!=test.tolist():
                raise ValueError("Resume split mismatch")
        else:
            audit=[]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                # Optimizer nonconvergence is a failed fit, never a fake AUC.
                warnings.simplefilter("error",ConvergenceWarning)
                pred=outer_fold(data,y,train,test,cfg,cfg["seed"]+10000*repeat+fold,audit)
            record={"train":train.tolist(),"test":test.tolist(),"audit":audit,
                    "predictions":{p:v.tolist() for p,v in pred.items()},
                    "warnings":[str(w.message) for w in caught]}
            for p,v in pred.items():
                if v.shape!=(len(test),) or not np.isfinite(v).all() or (v<0).any() or (v>1).any():
                    raise ValueError(f"Invalid probability output {p}")
            atomic_json(path,record)
        empty_count+=sum(a["empty"] for a in record["audit"])
        for p in MODELS:
            probabilities[p][test,repeat]=record["predictions"][p]
        atomic_json(directory.parent/"status.json",{"unit":directory.name,"repeat":repeat+1,
                    "fold":fold+1,"state":"running"})
    frame=pd.DataFrame({"subject_id":data.ids,"label":y})
    auc={}
    for p,v in probabilities.items():
        if not np.isfinite(v).all():
            raise ValueError("Incomplete outer OOF predictions")
        frame[p]=v.mean(axis=1)
        auc[p]=float(roc_auc_score(y,frame[p]))
    frame.to_csv(directory/"subject_oof.csv",index=False)
    result={"auc":auc,"empty_feature_events":empty_count,"state":"complete"}
    atomic_json(directory/"complete.json",result)
    return result


def permutation_p(observed,null):
    null=np.asarray(null,float)
    if not np.isfinite(null).all() or not len(null):
        raise ValueError("Complete finite null statistics required")
    return float((1+np.count_nonzero(null>=observed))/(len(null)+1))


def protocol(root,cfg):
    source=Path(__file__).parent
    return {"config":cfg,"dataset_hashes":fingerprint(root),
            "source_hashes":{p.name:file_hash(p) for p in sorted(source.glob("*.py"))},
            "python":platform.python_version(),"platform":platform.platform(),
            "packages":{n:importlib.metadata.version(n) for n in
                        ("numpy","scipy","pandas","scikit-learn","torch","mne")}}


def run(root,config,output,stage):
    cfg=json.loads(Path(config).read_text(encoding="utf-8"))
    for key in ("folds","repeats","inner_folds","permutations","pilots"):
        if not isinstance(cfg[key],int) or cfg[key]<1:
            raise ValueError(f"Positive integer required: {key}")
    if cfg["folds"]<2 or cfg["inner_folds"]<2:
        raise ValueError("CV requires >=2 folds")
    data=load_dataset(root)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    lock=out/"RUNNING.lock"
    # Never silently break a stale lock: check the old process before manual removal.
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        current=protocol(root,cfg)
        saved=out/"protocol.json"
        if saved.exists() and json.loads(saved.read_text(encoding="utf-8"))!=current:
            raise ValueError("Code/data/environment/config changed: use a NEW output folder")
        atomic_json(saved,current)
        observed=full_statistic(data,data.y,cfg,out/"observed")
        if stage=="validate":
            # Fresh real-label repeat, not reuse of fold artifacts.
            second=full_statistic(data,data.y,cfg,out/"observed_repeat")
            a=pd.read_csv(out/"observed/subject_oof.csv");b=pd.read_csv(out/"observed_repeat/subject_oof.csv")
            error=float(np.max(np.abs(a[list(MODELS)].to_numpy()-b[list(MODELS)].to_numpy())))
            if error>cfg.get("reproduction_tolerance",1e-7):
                raise ValueError(f"Real-label repeatability failed: {error}")
            for i in range(cfg["pilots"]):
                rng=np.random.default_rng(seed_for(cfg["permutation_seed"]+1,i))
                full_statistic(data,rng.permutation(data.y),cfg,out/f"pilot_{i:04d}")
            atomic_json(out/"validation_complete.json",{"max_probability_difference":error,
                        "pilots":cfg["pilots"],"state":"complete"})
        elif stage=="formal":
            if not (out/"validation_complete.json").exists():
                raise ValueError("Run validate successfully before formal permutations")
            rows=[]
            for i in range(cfg["permutations"]):
                rng=np.random.default_rng(seed_for(cfg["permutation_seed"],i))
                result=full_statistic(data,rng.permutation(data.y),cfg,out/f"perm_{i:04d}")
                rows.append({"permutation":i,"state":"complete",**result["auc"],
                             "empty_feature_events":result["empty_feature_events"]})
                pd.DataFrame(rows).to_csv(out/"permutation_auc.csv",index=False)
            frame=pd.DataFrame(rows)
            summary=[{"model":p,"observed_auc":observed["auc"][p],
                      "n_permutations":len(frame),"p_upper":permutation_p(observed["auc"][p],frame[p])}
                     for p in MODELS]
            pd.DataFrame(summary).to_csv(out/"permutation_summary.csv",index=False)
        else:
            raise ValueError("stage must be validate or formal")
        atomic_json(out/"status.json",{"state":"complete","stage":stage})
    except Exception:
        with (out/"errors.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps({"stage":stage,"traceback":traceback.format_exc()})+"\n")
        atomic_json(out/"status.json",{"state":"failed","stage":stage})
        raise
    finally:
        lock.unlink(missing_ok=True)
