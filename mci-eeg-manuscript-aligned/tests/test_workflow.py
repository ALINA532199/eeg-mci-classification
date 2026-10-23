import json,tempfile,unittest,inspect,warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from mci_eeg.data import load_dataset
from mci_eeg.demo import make_demo
from mci_eeg.workflow import run,permutation_p,splits
from mci_eeg.selection import Selector,bh,cohen_d
from mci_eeg.features import extract,dwpli_squared,power_db
from mci_eeg.models import outer_fold
ROOT=Path(__file__).resolve().parents[1]


def small_features():
    f=json.loads((ROOT/'configs/preprocessing_example.json').read_text())['features']
    f['erp']=f['erp'][:2];f['tf']=f['tf'][:1]
    f['tf'][0]['all_frequencies']=f['tf'][0]['frequencies']
    f['connectivity']=f['connectivity'][:1];f['pac']=f['pac'][:1];f['pac'][0]['surrogates']=20
    return f


class MethodsTests(unittest.TestCase):
    def test_cohen_d(self):self.assertEqual(cohen_d([1,2,3],[3,4,5]),-2)
    def test_bh(self):np.testing.assert_allclose(bh([.01,.04,.03]),[.03,.04,.04])
    def test_permutation_ties(self):self.assertEqual(permutation_p(.7,[.5,.7,.8]),.75)
    def test_dwpli_negative_preserved(self):self.assertLess(dwpli_squared([-1,1,-1,1]),0)
    def test_selection_seed_and_family(self):
        x=np.random.default_rng(5).normal(size=(24,3));y=np.tile([0,1],12);x[:,0]+=y*5
        a=Selector(['ERP','TF','TF'],permutations=99,seed=2).fit(x,y)
        b=Selector(['ERP','TF','TF'],permutations=99,seed=2).fit(x,y)
        np.testing.assert_array_equal(a.p,b.p);self.assertEqual(a.p[0],.01)
    def test_repeated_subject_splits(self):
        y=np.tile([0,1],20);seen=np.zeros((40,2))
        for r,f,tr,te in splits(y,dict(seed=42,folds=5,repeats=2)):
            self.assertFalse(set(tr)&set(te));seen[te,r]+=1
        np.testing.assert_array_equal(seen,np.ones((40,2)))
    def test_short_window_features_documented(self):
        f=small_features();channels=[c for group in f['rois'].values() for c in group]
        times=np.arange(-.5,1+1/500,1/500);e=np.random.default_rng(2).normal(scale=1e-6,size=(12,12,len(times)))
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always');values,schema=extract(e,times,500,channels,f)
        self.assertTrue(np.isfinite(list(values.values())).all())
        self.assertEqual({v['modality'] for v in schema.values()},{'ERP','TF','PAC','dwPLI'})
        self.assertTrue(any('sparse phase' in str(w.message) for w in messages))
        self.assertTrue(any('correlated' in str(w.message) for w in messages))
    def test_model_test_label_invariance_and_shared_behavior(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'data';make_demo(root);d=load_dataset(root)
            cfg=json.loads((ROOT/'configs/smoke.json').read_text());tr=np.arange(16);te=np.arange(16,24)
            original=d.x.copy();audit=[];a=outer_fold(d,d.y,tr,te,cfg,42,audit)
            y=d.y.copy();y[te]=1-y[te];b=outer_fold(d,y,tr,te,cfg,42,[])
            for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-7)
            np.testing.assert_array_equal(d.x,original)
            common=[v for v in audit if v['block']=='common'];self.assertEqual(len(common),1)
            self.assertIn('measured_rt',common[0]['selected']);self.assertEqual(common[0]['shared_by'],['P3','P6'])
            deep=next(v['training'] for v in audit if v['block']=='deep')
            self.assertFalse(set(deep['train_subject_positions'])&set(deep['validation_subject_positions']))
    def test_full_workflow_and_resume(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);make_demo(root/'data');cfg=ROOT/'configs/smoke.json'
            run(root/'data',cfg,root/'out','validate');run(root/'data',cfg,root/'out','formal')
            result=(root/'out/permutation_summary.csv').read_bytes()
            run(root/'data',cfg,root/'out','formal');self.assertEqual(result,(root/'out/permutation_summary.csv').read_bytes())
            self.assertEqual(len(pd.read_csv(root/'out/permutation_summary.csv')),6)
    def test_failure_not_imputed(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);make_demo(root/'data')
            with patch('mci_eeg.workflow.outer_fold',side_effect=RuntimeError('test failure')):
                with self.assertRaises(RuntimeError):run(root/'data',ROOT/'configs/smoke.json',root/'out','validate')
            self.assertTrue((root/'out/errors.jsonl').exists());self.assertFalse((root/'out/observed/complete.json').exists())
    def test_ica_review_and_mastoid_reference(self):
        import mne
        from mci_eeg.preprocess import review_ica,prepare,load_referenced
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);rng=np.random.default_rng(46)
            cfg=json.loads((ROOT/'configs/preprocessing_example.json').read_text());cfg['features']=small_features()
            cfg['minimum_trials']=3;cfg['sfreq']=128;cfg['ica_components']=.95
            scalp=cfg['eegnet_channels']+['Cz'];channels=scalp+['M1','M2']
            for i in range(2):
                sources=rng.laplace(scale=1e-6,size=(len(channels),128*70))
                raw=mne.io.RawArray(sources,mne.create_info(channels,128,'eeg'),verbose=False)
                raw.set_annotations(mne.Annotations([10,20,30,40,50,60],[0]*6,['probe']*6))
                raw.save(root/f'synthetic_{i}_raw.fif',overwrite=True,verbose=False)
            rows=pd.DataFrame([dict(subject_id=f'synthetic_{i}',clinical_label=i,raw_file=f'synthetic_{i}_raw.fif',bad_channels='[]',bad_segments='[]') for i in range(2)])
            rows.to_csv(root/'manifest.csv',index=False)
            pd.DataFrame([dict(subject_id=f'synthetic_{i}',ACC=.8,RT_seconds=1.,IES_seconds=1.25) for i in range(2)]).to_csv(root/'behavior.csv',index=False)
            (root/'cfg.json').write_text(json.dumps(cfg))
            # Directly verify subtraction of the mastoid average before filtering.
            nofilter={**cfg,'notch_hz':[]}
            original=mne.io.read_raw_fif(root/'synthetic_0_raw.fif',preload=True,verbose=False)
            referenced,picks=load_referenced(root/'synthetic_0_raw.fif',nofilter,next(rows.itertuples()))
            expected=original.get_data(picks=['F3'])-original.get_data(picks=['M1','M2']).mean(axis=0)
            np.testing.assert_allclose(referenced.get_data(picks=['F3']),expected,atol=1e-15)
            review_ica(root/'manifest.csv',root/'cfg.json',root/'review')
            # Synthetic approvals are test fixtures only, never clinical reviews.
            for i in range(2):
                p=root/'review'/f'subject_{i:05d}'/'review.json';r=json.loads(p.read_text());r.update(approved=True,reviewer='synthetic_test_fixture');p.write_text(json.dumps(r))
            prepare(root/'manifest.csv',root/'behavior.csv',root/'cfg.json',root/'prepared',root/'review')
            d=load_dataset(root/'prepared');self.assertEqual(d.epochs[0].shape[1],12)
            audit=json.loads((root/'prepared/preprocessing_audit.json').read_text())
            self.assertEqual(len(audit[0]['ica_channels']),13);self.assertNotIn('M1',audit[0]['ica_channels'])
            self.assertTrue((root/'review/subject_00000/report.html').exists())
            # Editing preprocessing after review must invalidate the binding.
            cfg['highpass']=.2;(root/'cfg.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'binding changed'):
                prepare(root/'manifest.csv',root/'behavior.csv',root/'cfg.json',root/'bad_prepare',root/'review')
    def test_rng_compatibility(self):
        from mci_eeg.preprocess import make_ica
        captured={}
        def fake(n_components=None,method=None,max_iter=None,rng=None):captured.update(rng=rng);return captured
        with patch('mne.preprocessing.ICA',fake):
            make_ica({'ica_components':.99,'seed':42})
        self.assertEqual(captured['rng'],42)
    def test_shap_stacking(self):
        from mci_eeg.explain import explain
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);make_demo(root/'data');explain(root/'data',ROOT/'configs/smoke.json',root/'explain')
            self.assertTrue((root/'explain/P6_posthoc_importance.csv').exists())
            self.assertTrue((root/'explain/P6_posthoc_summary.png').exists())


if __name__=='__main__':unittest.main()
