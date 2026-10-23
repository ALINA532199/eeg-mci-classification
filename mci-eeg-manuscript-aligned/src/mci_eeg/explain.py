"""Separate full-data stacking interpretation, never OOF performance estimates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .data import load_dataset
from .selection import bh
from .deep import DeepFit
from .models import common_inputs,stack_fit_predict
from .workflow import atomic_json


def explain(root,config,output):
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=load_dataset(root);cfg=json.loads(Path(config).read_text(encoding='utf-8-sig'))
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    indices=np.arange(len(d.y));audit=[]
    common=common_inputs(d,d.y,indices,indices,cfg,cfg['seed'],audit)['common'][0]
    names=next(a['selected'] for a in audit if a['block']=='common')
    deep=DeepFit(cfg['deep'],cfg['seed']).fit(d.epochs,d.y)
    _,z=deep.transform(d.epochs);z=StandardScaler().fit_transform(z)
    pca=PCA(n_components=cfg['pca_components'],svd_solver='full');pc=pca.fit_transform(z)
    full=np.column_stack((common,pc))
    for name,x,columns in [('P3_posthoc',common,names),('P6_posthoc',full,names+[f'EEGNet_PC{i+1}' for i in range(pc.shape[1])])]:
        predict=stack_fit_predict(x,d.y,x,cfg,cfg['seed'],True,return_predictor=True)
        background=x[np.random.default_rng(cfg['seed']).choice(len(x),min(30,len(x)),replace=False)]
        explainer=shap.Explainer(predict,shap.maskers.Independent(background),algorithm='permutation',seed=cfg['seed'])
        explanation=explainer(x,max_evals=2*x.shape[1]+1)
        importance=np.abs(explanation.values).mean(axis=0)
        frame=pd.DataFrame({'feature':columns,'mean_absolute_SHAP_probability':importance,
                            'contribution_percent':100*importance/max(importance.sum(),1e-12)})
        frame.sort_values('mean_absolute_SHAP_probability',ascending=False).to_csv(out/f'{name}_importance.csv',index=False)
        import inspect
        plot_options={"rng":np.random.default_rng(cfg["seed"])} if "rng" in inspect.signature(shap.summary_plot).parameters else {}
        shap.summary_plot(explanation.values,x,feature_names=columns,max_display=10,show=False,**plot_options)
        plt.savefig(out/f'{name}_summary.png',dpi=160,bbox_inches='tight');plt.close('all')
        top=np.argsort(-importance)[:10]
        fig,ax=plt.subplots();ax.barh(np.asarray(columns)[top][::-1],importance[top][::-1]);ax.set_xlabel('Mean absolute SHAP (probability)')
        fig.savefig(out/f'{name}_bar.png',dpi=160,bbox_inches='tight');plt.close(fig)
        correlations=[]
        for j,a in enumerate(top):
            for b in top[j+1:]:
                rho,p=spearmanr(x[:,a],x[:,b])
                if not np.isfinite(p):continue
                correlations.append({'a':columns[a],'b':columns[b],'rho':float(rho),'p':float(p)})
        if correlations:
            cf=pd.DataFrame(correlations);cf['q']=bh(cf.p);cf.to_csv(out/f'{name}_top_spearman.csv',index=False)
    atomic_json(out/'interpretation_scope.json',{'scope':'Separate full-data post-hoc stacking models, NOT OOF explanations',
        'embedding':'One fitted network; no averaging of independently trained latent coordinate systems',
        'pca_explained_variance':pca.explained_variance_ratio_.tolist(),'shap':'Permutation approximation in probability units',
        'correlation_tests':'All pairs among up to ten highest-ranked inputs; family BH'})
