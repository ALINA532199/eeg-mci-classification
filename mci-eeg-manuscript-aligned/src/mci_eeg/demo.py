"""Artificial signals solely for software tests; never clinical results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def make_demo(output,n=24):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/"epochs").mkdir()
    rng=np.random.default_rng(17)
    y=np.tile([0,1],n//2)
    if len(y)!=n:
        raise ValueError("Demo n must be even")
    records,features=[],[]
    for i,label in enumerate(y):
        sid=f"synthetic_{i:04d}"
        e=rng.normal(size=(4,3,128)).astype(np.float32)
        e[:,0]+=label*0.3*np.sin(np.arange(128)*0.2)
        path=f"epochs/{sid}.npz"
        np.savez_compressed(out/path,epochs=e,sfreq=128.,channels=["Fz","Pz","Oz"])
        records.append({"subject_id":sid,"label":int(label),"epochs_file":path})
        features.append({"subject_id":sid,"measured_rt":float(rng.normal()+label*0.5),
                         "ERP_P3":float(rng.normal()+label*0.5),"TF_theta":float(rng.normal())})
    pd.DataFrame(records).to_csv(out/"subjects.csv",index=False)
    pd.DataFrame(features).to_csv(out/"features.csv",index=False)
    (out/"feature_schema.json").write_text(json.dumps({"measured_rt":"Behavior","ERP_P3":"ERP","TF_theta":"TF"}),encoding="utf-8")
    (out/"dataset.json").write_text(json.dumps({"label_source":"synthetic_demo","warning":"NO PATIENT DATA; NOT SCIENTIFIC EVIDENCE"}),encoding="utf-8")
