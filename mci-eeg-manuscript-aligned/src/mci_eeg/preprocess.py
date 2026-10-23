"""Full-scalp mastoid-referenced ICA with reusable, label-blind review artifacts."""
from pathlib import Path
import inspect,json,html,warnings
import numpy as np
import pandas as pd
import mne
from .features import extract,window_mask
from .workflow import atomic_json
from .data import file_hash


def _raw_path(manifest,row):return (Path(manifest).parent/row.raw_file).resolve()


def source_hashes(path):
    files=[Path(path)]
    if Path(path).suffix.lower()=='.set' and Path(path).with_suffix('.fdt').exists():files.append(Path(path).with_suffix('.fdt'))
    return {p.name:file_hash(p) for p in files}


def load_referenced(path,cfg,row):
    reader=mne.io.read_raw_eeglab if path.suffix.lower()=='.set' else mne.io.read_raw_fif
    if path.suffix.lower() not in {'.set','.fif'}:raise ValueError('Continuous SET/FIF required')
    raw=reader(path,preload=True,verbose=False)
    aux={ch:kind for kind,key in [('eog','eog_channels'),('ecg','ecg_channels')] for ch in cfg.get(key,[])}
    if aux:raw.set_channel_types(aux)
    refs=cfg['reference_channels']
    if refs!=['M1','M2']:raise ValueError('This protocol uses the average of M1 and M2')
    if any(ch not in raw.ch_names for ch in refs):raise ValueError('M1 and M2 must be recorded; missing references cannot be reconstructed')
    scalp=cfg.get('scalp_channels') or [ch for ch,typ in zip(raw.ch_names,raw.get_channel_types()) if typ=='eeg' and ch not in refs]
    if set(scalp)&set(refs) or len(set(scalp))!=len(scalp):raise ValueError('Scalp and reference channel sets must be distinct')
    required=set(cfg['eegnet_channels'])|{c for group in cfg['features']['rois'].values() for c in group}
    if not required.issubset(scalp):raise ValueError('All ROI/EEGNet channels must be in the scalp montage')
    raw.pick(list(dict.fromkeys(scalp+refs+list(aux))))
    if cfg.get('montage'):raw.set_montage(cfg['montage'],on_missing='raise',verbose=False)
    bad=json.loads(getattr(row,'bad_channels','[]'))
    if set(bad)&set(refs):raise ValueError('Bad mastoid reference requires manual resolution')
    raw.info['bads']=bad
    if bad:raw.interpolate_bads(reset_bads=True,verbose=False)
    segments=json.loads(getattr(row,'bad_segments','[]'))
    for onset,duration in segments:raw.annotations.append(onset,duration,'BAD_manual')
    raw.set_eeg_reference(ref_channels=refs,projection=False,verbose=False)
    if cfg.get('notch_hz'):
        raw.notch_filter(cfg['notch_hz'],notch_widths=cfg.get('notch_width_hz',4),picks=scalp,verbose=False)
    # Keep the full montage for ICA; ROI and EEGNet selection happens later.
    return raw,scalp


def make_ica(cfg):
    options=dict(n_components=cfg['ica_components'],method=cfg.get('ica_method','fastica'),max_iter=cfg.get('ica_max_iter',1000))
    if 'rng' in inspect.signature(mne.preprocessing.ICA).parameters:
        options['rng']=cfg['seed']
    else:options['random_state']=cfg['seed']
    return mne.preprocessing.ICA(**options)


def review_ica(manifest,config,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=Path(manifest);cfg=json.loads(Path(config).read_text(encoding='utf-8-sig'))
    rows=pd.read_csv(manifest,dtype={'subject_id':str})
    if rows.subject_id.isna().any() or rows.subject_id.duplicated().any():raise ValueError('Unique IDs required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for number,row in enumerate(rows.itertuples()):
        folder=out/f'subject_{number:05d}';folder.mkdir()
        path=_raw_path(manifest,row);raw,scalp=load_referenced(path,cfg,row)
        fit_raw=raw.copy().pick(scalp).filter(cfg.get('ica_fit_highpass',1.),cfg['lowpass'],verbose=False)
        ica=make_ica(cfg);ica.fit(fit_raw,picks=scalp,reject_by_annotation=True,verbose=False)
        if ica.n_iter_>=cfg.get('ica_max_iter',1000):raise RuntimeError('ICA did not converge')
        ica.save(folder/'decomposition-ica.fif',overwrite=False,verbose=False)
        plots=ica.plot_components(show=False)
        if not isinstance(plots,list):plots=[plots]
        images=[]
        for j,fig in enumerate(plots):
            name=f'components_{j:02d}.png';fig.savefig(folder/name,dpi=130);plt.close(fig);images.append(name)
        sources=ica.get_sources(fit_raw).get_data()
        for start in range(0,len(sources),8):
            fig,axes=plt.subplots(min(8,len(sources)-start),1,figsize=(12,10),squeeze=False)
            n=min(sources.shape[1],int(raw.info['sfreq']*20))
            for j,ax in enumerate(axes[:,0]):
                ax.plot(np.arange(n)/raw.info['sfreq'],sources[start+j,:n],lw=.4);ax.set_ylabel(f'IC {start+j}')
            axes[-1,0].set_xlabel('Time (s), first 20 seconds')
            fig.tight_layout();name=f'sources_{start:03d}.png';fig.savefig(folder/name,dpi=120);plt.close(fig);images.append(name)
        fig,ax=plt.subplots(figsize=(12,5))
        from scipy.signal import welch
        frequencies,psd=welch(sources,fs=raw.info['sfreq'],nperseg=min(1024,sources.shape[1]),axis=-1)
        for j,spectrum in enumerate(psd):ax.semilogy(frequencies,spectrum,label=f'IC {j}')
        ax.set_xlim(0,cfg['lowpass']);ax.legend(ncol=6,fontsize=6);fig.tight_layout()
        fig.savefig(folder/'spectra.png',dpi=130);plt.close(fig);images.append('spectra.png')
        suggestions={}
        for key,method in [('eog_channels','find_bads_eog'),('ecg_channels','find_bads_ecg')]:
            for channel in cfg.get(key,[]):
                try:
                    options={'ch_name':channel,'verbose':False}
                    if key=='ecg_channels':options['method']='correlation'
                    indices,scores=getattr(ica,method)(raw,**options)
                    suggestions[channel]={'component_candidates':list(map(int,indices)),'scores':np.asarray(scores).tolist()}
                except (ValueError,RuntimeError) as error:
                    suggestions[channel]={'unavailable':str(error)}
        atomic_json(folder/'binding.json',{'subject_id':str(row.subject_id),'raw_hashes':source_hashes(path),
            'config':cfg,'mne_version':mne.__version__,'ica_hash':file_hash(folder/'decomposition-ica.fif'),
            'cleaning_spec':{'bad_channels':getattr(row,'bad_channels','[]'),'bad_segments':getattr(row,'bad_segments','[]')},
            'ica_channels':ica.ch_names,'rng_api':'rng' if 'rng' in inspect.signature(mne.preprocessing.ICA).parameters else 'random_state'})
        atomic_json(folder/'review.json',{'subject_id':str(row.subject_id),'approved':False,'reviewer':'',
            'exclude':{'ocular':[],'saccade':[],'cardiac':[],'other':[]},'notes':''})
        atomic_json(folder/'artifact_candidates.json',suggestions)
        body='<h1>ICA review: '+html.escape(str(row.subject_id))+'</h1><p>Review ocular, saccadic and cardiac artifacts. Candidate scores are advisory. Edit review.json after inspecting component maps, sources and spectra. Diagnostic labels are omitted.</p>'
        body+=''.join('<img style="max-width:100%" src="'+name+'">' for name in images)
        (folder/'report.html').write_text('<!doctype html><meta charset="utf-8">'+body,encoding='utf-8')
    atomic_json(out/'index.json',[{'subject_id':str(row.subject_id),'folder':f'subject_{i:05d}'} for i,row in enumerate(rows.itertuples())])


def prepare(manifest,behavior,config,output,ica_directory):
    manifest=Path(manifest);cfg=json.loads(Path(config).read_text(encoding='utf-8-sig'))
    rows=pd.read_csv(manifest,dtype={'subject_id':str})
    beh=pd.read_csv(behavior,dtype={'subject_id':str}).set_index('subject_id')
    if rows.subject_id.isna().any() or rows.subject_id.duplicated().any() or set(rows.clinical_label)!={0,1}:raise ValueError('Unique subjects and binary clinical labels required')
    if beh.index.duplicated().any() or set(beh.index)!=set(rows.subject_id) or list(beh.columns)!=cfg['behavior_columns']:raise ValueError('Behavior mapping/columns mismatch')
    if not np.isfinite(beh.to_numpy(float)).all():raise ValueError('Finite measured behavior required')
    review_root=Path(ica_directory)
    index={r['subject_id']:r['folder'] for r in json.loads((review_root/'index.json').read_text())}
    out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'epochs').mkdir()
    features=[];subjects=[];audits=[];schema={n:{'modality':'Behavior','testing_family':'Behavior'} for n in beh.columns}
    for number,row in enumerate(rows.itertuples()):
        folder=review_root/index[str(row.subject_id)];binding=json.loads((folder/'binding.json').read_text())
        review=json.loads((folder/'review.json').read_text())
        path=_raw_path(manifest,row)
        if binding['config']!=cfg or binding['raw_hashes']!=source_hashes(path) or binding['mne_version']!=mne.__version__:raise ValueError('ICA review binding changed; regenerate and review')
        if binding['cleaning_spec']!={'bad_channels':getattr(row,'bad_channels','[]'),'bad_segments':getattr(row,'bad_segments','[]')}:raise ValueError('Manual cleaning changed after ICA review')
        if binding['ica_hash']!=file_hash(folder/'decomposition-ica.fif'):raise ValueError('ICA file changed')
        if review.get('approved') is not True or not review.get('reviewer','').strip() or review['subject_id']!=str(row.subject_id):raise ValueError('Completed label-blind ICA review required')
        raw,scalp=load_referenced(path,cfg,row)
        ica=mne.preprocessing.read_ica(folder/'decomposition-ica.fif',verbose=False)
        exclude=sorted(set(i for group in review['exclude'].values() for i in group))
        if any(type(i)!=int or not 0<=i<ica.n_components_ for i in exclude):raise ValueError('Invalid excluded component')
        raw.filter(cfg['highpass'],cfg['lowpass'],picks=scalp,verbose=False)
        ica.apply(raw,exclude=exclude,verbose=False)
        events,event_id=mne.events_from_annotations(raw,event_id=cfg['event_id'],verbose=False)
        ep=mne.Epochs(raw,events,event_id=event_id,tmin=cfg['epoch'][0],tmax=cfg['epoch'][1],baseline=tuple(cfg['baseline']),
            picks=scalp,preload=True,reject_by_annotation=True,event_repeated='error',verbose=False)
        if cfg.get('sfreq') is not None and cfg['sfreq']!=ep.info['sfreq']:ep.resample(cfg['sfreq'],verbose=False)
        # Manuscript specifies absolute +/-80 uV, not 160 uV peak-to-peak.
        rejected=np.flatnonzero(np.max(np.abs(ep.get_data()),axis=(1,2))>cfg['reject_absolute_uv']*1e-6)
        ep.drop(rejected,reason='ABSOLUTE_AMPLITUDE',verbose=False)
        if len(ep)<cfg['minimum_trials']:raise ValueError(f'Insufficient retained trials: row {number}')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            vals,fschema=extract(ep.get_data(copy=True),ep.times,ep.info['sfreq'],ep.ch_names,cfg['features'])
        schema.update(fschema)
        features.append({'subject_id':row.subject_id,**beh.loc[row.subject_id].to_dict(),**vals})
        deep=ep.copy().pick(cfg['eegnet_channels'])
        array=deep.get_data()[:,:,window_mask(deep.times,cfg['eegnet_window'])].astype(np.float32)
        rel=f'epochs/subject_{number:05d}.npz'
        np.savez_compressed(out/rel,epochs=array,sfreq=deep.info['sfreq'],channels=deep.ch_names)
        subjects.append({'subject_id':row.subject_id,'label':int(row.clinical_label),'epochs_file':rel})
        audits.append({'subject_id':row.subject_id,'reference_channels':cfg['reference_channels'],'ica_channels':ica.ch_names,
            'eegnet_channels':deep.ch_names,'retained_trials':len(ep),'ica_review':review,'raw_hashes':source_hashes(path),
            'ica_hash':binding['ica_hash'],'feature_warnings':[str(w.message) for w in caught],'drop_log':[list(v) for v in ep.drop_log]})
    pd.DataFrame(subjects).to_csv(out/'subjects.csv',index=False);pd.DataFrame(features).to_csv(out/'features.csv',index=False)
    atomic_json(out/'feature_schema.json',schema);atomic_json(out/'preprocessing_audit.json',audits)
    atomic_json(out/'preprocessing_config.json',cfg)
    atomic_json(out/'dataset.json',{'label_source':'independent_clinical_diagnosis'})
