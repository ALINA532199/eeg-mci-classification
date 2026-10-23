"""Outer-training representations and inner stacking, with P3/P6 passthrough.
Inner predictions condition on the outer-training representation; only outer OOF
predictions estimate generalization of the full pipeline.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from .selection import Selector
from .deep import DeepFit
MODELS=("P1","P2","P3","P4","P5","P6")


def base_models(cfg,seed):
    return [LogisticRegression(C=cfg["C"],max_iter=2000,random_state=seed),
            SVC(C=cfg["C"],kernel="linear",probability=True,random_state=seed),
            SVC(C=cfg["C"],kernel="rbf",probability=True,random_state=seed),
            RandomForestClassifier(n_estimators=cfg["trees"],max_depth=cfg["max_depth"],random_state=seed,n_jobs=1),
            GradientBoostingClassifier(n_estimators=cfg["boosting_trees"],random_state=seed)]


def fit_base(x,y,test,cfg,seed):
    return np.column_stack([m.fit(x,y).predict_proba(test)[:,1] for m in base_models(cfg,seed)])


def stack_fit_predict(x,y,test,cfg,seed,passthrough,return_predictor=False):
    if not x.shape[1]:return np.full(len(test),y.mean())
    probabilities=np.full((len(y),5),np.nan)
    cv=StratifiedKFold(cfg["inner_folds"],shuffle=True,random_state=seed)
    for inner,(tr,va) in enumerate(cv.split(x,y)):
        probabilities[va]=fit_base(x[tr],y[tr],x[va],cfg["base"],seed+inner)
    fitted=[m.fit(x,y) for m in base_models(cfg["base"],seed)]
    external=np.column_stack([m.predict_proba(test)[:,1] for m in fitted])
    if passthrough:
        probabilities=np.column_stack((probabilities,x))
        external=np.column_stack((external,test))
    scale=StandardScaler().fit(probabilities)
    meta=LogisticRegression(C=cfg["meta_C"],max_iter=2000,random_state=seed)
    meta.fit(scale.transform(probabilities),y)
    def predict(matrix):
        base=np.column_stack([m.predict_proba(matrix)[:,1] for m in fitted])
        inputs=np.column_stack((base,matrix)) if passthrough else base
        return meta.predict_proba(scale.transform(inputs))[:,1]
    return predict if return_predictor else meta.predict_proba(scale.transform(external))[:,1]


def common_inputs(data,y,train,test,cfg,seed,audit):
    families=np.asarray(data.families)
    testing=np.asarray(data.testing_families if data.testing_families is not None else data.families)
    beh=families=="Behavior";eeg=~beh
    if not beh.any() or not eeg.any():raise ValueError("Behavior and EEG inputs required")
    blocks={}
    for name,mask,method in (("behavior",beh,cfg.get("behavior_test","welch")),("eeg",eeg,"permutation")):
        options={**cfg["selection"],"method":method,"seed":seed,"testing_families":testing[mask]}
        selector=Selector(families[mask],**options).fit(data.x[train][:,mask],y[train])
        blocks[name]=(selector.transform(data.x[train][:,mask]),selector.transform(data.x[test][:,mask]))
        audit.append({"context":"outer","block":name,"empty":len(selector.indices)==0,
                      "selected":np.asarray(data.names)[mask][selector.indices].tolist(),
                      "n_train":len(train),"n_test":len(test),"test_method":method})
    bs=StandardScaler().fit(data.x[train][:,beh])
    blocks["common"]=(np.column_stack((bs.transform(data.x[train][:,beh]),blocks["eeg"][0])),
                      np.column_stack((bs.transform(data.x[test][:,beh]),blocks["eeg"][1])))
    audit.append({"context":"outer","block":"common","empty":False,
                  "selected":np.asarray(data.names)[beh].tolist()+audit[-1]["selected"],
                  "behavior_policy":"retain_all","shared_by":["P3","P6"]})
    return blocks


def outer_fold(data,y,train,test,cfg,seed,audit):
    if np.bincount(y[train],minlength=2).min()<cfg["inner_folds"]:
        raise ValueError("Too few subjects per class for inner CV")
    blocks=common_inputs(data,y,train,test,cfg,seed,audit)
    deep=DeepFit(cfg["deep"],seed).fit([data.epochs[i] for i in train],y[train])
    _,z=deep.transform([data.epochs[i] for i in train])
    p5,zt=deep.transform([data.epochs[i] for i in test])
    scaler=StandardScaler().fit(z)
    k=cfg["pca_components"]
    if k>min(len(train)-1,z.shape[1]):raise ValueError("Insufficient data for prespecified PCA dimension")
    pca=PCA(n_components=k,svd_solver="full").fit(scaler.transform(z))
    blocks["fusion"]=(np.column_stack((blocks["common"][0],pca.transform(scaler.transform(z)))),
                      np.column_stack((blocks["common"][1],pca.transform(scaler.transform(zt)))))
    audit.append({"context":"outer","block":"deep","empty":False,"training":deep.training_audit})
    result={"P5":p5}
    for p,name in (("P1","behavior"),("P2","eeg"),("P3","common"),("P6","fusion")):
        a,b=blocks[name]
        result[p]=stack_fit_predict(a,y[train],b,cfg,seed,passthrough=p in {"P3","P6"})
    a,b=blocks["eeg"]
    result["P4"]=(MLPClassifier(hidden_layer_sizes=tuple(cfg["mlp"]["hidden"]),max_iter=cfg["mlp"]["max_iter"],
        random_state=seed,early_stopping=False).fit(a,y[train]).predict_proba(b)[:,1]
        if a.shape[1] else np.full(len(test),y[train].mean()))
    return result
