"""One row per independently diagnosed subject. No label generation from EEG."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


@dataclass
class Dataset:
    ids: np.ndarray
    y: np.ndarray
    x: np.ndarray
    names: list
    families: list
    epochs: list
    sfreq: float
    channels: list
    testing_families: list = None


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_dataset(root):
    root = Path(root)
    meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if meta.get("label_source") not in {"independent_clinical_diagnosis", "synthetic_demo"}:
        raise ValueError("Declare independent clinical labels or explicitly synthetic demo labels")
    subjects = pd.read_csv(root / "subjects.csv", dtype={"subject_id": str})
    features = pd.read_csv(root / "features.csv", dtype={"subject_id": str})
    for frame in (subjects, features):
        if frame.subject_id.isna().any() or frame.subject_id.duplicated().any():
            raise ValueError("subject_id must be globally unique, nonmissing, one row per person")
    if set(subjects.subject_id) != set(features.subject_id):
        raise ValueError("Subject and feature rows must match exactly")
    if set(subjects.label) != {0, 1}:
        raise ValueError("Clinical label must be 0 or 1, both classes required")
    schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
    names = list(schema)
    if set(features.columns) != {"subject_id", *names}:
        raise ValueError("Features must exactly match explicitly whitelisted schema")
    modalities={k:(v["modality"] if isinstance(v,dict) else v) for k,v in schema.items()}
    tests={k:(v.get("testing_family",v["modality"]) if isinstance(v,dict) else v) for k,v in schema.items()}
    if not names or any(v not in {"Behavior", "ERP", "TF", "PAC", "dwPLI"} for v in modalities.values()):
        raise ValueError("Only measured behavior / EEG predictors are allowed")
    x = features.set_index("subject_id").loc[subjects.subject_id, names].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features: resolve missingness with a prespecified policy")
    epochs, hashes, channels, sfreq, n_times = [], set(), None, None, None
    for row in subjects.itertuples():
        path = (root / row.epochs_file).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Epoch files must be contained in the private dataset folder")
        with np.load(path, allow_pickle=False) as z:
            e = z["epochs"].astype(np.float32)
            ch = z["channels"].tolist()
            fs = float(z["sfreq"])
        if e.ndim != 3 or len(e) < 2 or not np.isfinite(e).all() or e.shape[1] != len(ch):
            raise ValueError("Epochs must be finite trials x channels x time, >=2 trials")
        if e.shape[2] < 64 or fs <= 0 or len(set(ch)) != len(ch):
            raise ValueError("Invalid epoch geometry")
        if channels is None:
            channels, sfreq, n_times = ch, fs, e.shape[2]
        if channels != ch or sfreq != fs or n_times != e.shape[2]:
            raise ValueError("All subjects require identical channel order, time grid, sampling rate")
        digest = hashlib.sha256(e.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError("Identical EEG assigned to different subjects; audit source mapping")
        hashes.add(digest)
        epochs.append(e)
    return Dataset(subjects.subject_id.to_numpy(), subjects.label.to_numpy(int), x,
                   names, list(modalities.values()), epochs, sfreq, channels, list(tests.values()))


def fingerprint(root):
    root = Path(root)
    files = [root / name for name in ("dataset.json", "subjects.csv", "features.csv", "feature_schema.json")]
    files.extend(root / name for name in ("preprocessing_config.json","preprocessing_audit.json") if (root/name).exists())
    frame = pd.read_csv(root / "subjects.csv")
    files.extend(root / p for p in frame.epochs_file)
    return {str(p.relative_to(root)): file_hash(p) for p in files}
