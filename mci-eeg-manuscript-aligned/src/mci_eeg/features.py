"""Manuscript feature definitions with explicit short-window estimator diagnostics."""
import warnings
import numpy as np
from scipy.signal import butter,sosfiltfilt,hilbert
from mne.time_frequency import tfr_array_morlet


def window_mask(times,window):
    if window[0]>=window[1] or window[0]<times[0]-1e-6 or window[1]>times[-1]+1e-6:raise ValueError('Window outside epoch')
    mask=(times>=window[0]-1e-10)&(times<=window[1]+1e-10)
    if mask.sum()<3:raise ValueError('Insufficient window samples')
    return mask


def analytic(x,band,sfreq):
    if not 0<band[0]<band[1]<sfreq/2:raise ValueError('Invalid filter band')
    return hilbert(sosfiltfilt(butter(4,band,btype='bandpass',fs=sfreq,output='sos'),x,axis=-1),axis=-1)


def modulation_index(phase,amplitude,bins=18):
    p=np.ravel(phase);a=np.ravel(amplitude)
    ix=np.clip(((p+np.pi)/(2*np.pi)*bins).astype(int),0,bins-1)
    count=np.bincount(ix,minlength=bins)
    means=np.divide(np.bincount(ix,weights=a,minlength=bins),count,out=np.zeros(bins),where=count>0)
    if means.sum()<=0:raise ValueError('Undefined PAC: zero amplitude')
    dist=means/means.sum();nz=dist>0
    return float(np.sum(dist[nz]*np.log(dist[nz]*bins))/np.log(bins))


def dwpli_squared(imag):
    v=np.asarray(imag,float).ravel();squares=np.square(v).sum()
    denominator=np.abs(v).sum()**2-squares
    if denominator<=np.finfo(float).eps*max(squares,1e-300):raise ValueError('Undefined dwPLI2')
    return float((v.sum()**2-squares)/denominator)


def power_db(epochs,sfreq,frequencies,baseline_mask):
    """Average trial power first, then baseline-normalize per channel/frequency."""
    freq=np.asarray(frequencies,float);cycles=np.maximum(3,freq/2)
    pad=int(np.ceil(np.max(5*cycles/(2*np.pi*freq))*sfreq))+1
    sums=None
    for start in range(0,len(epochs),4):
        padded=np.pad(epochs[start:start+4],((0,0),(0,0),(pad,pad)),mode='reflect')
        power=tfr_array_morlet(padded,sfreq,freq,n_cycles=cycles,zero_mean=True,output='power',verbose=False)[...,pad:pad+epochs.shape[-1]]
        batch=power.sum(axis=0);sums=batch if sums is None else sums+batch
    power=sums/len(epochs)
    baseline=power[...,baseline_mask].mean(axis=-1,keepdims=True)
    if (baseline<=0).any():raise ValueError('Zero baseline power')
    return 10*np.log10(np.maximum(power,np.finfo(float).tiny)/baseline)


def extract(epochs,times,sfreq,channels,cfg):
    epochs=np.asarray(epochs,float)
    rois={name:epochs[:,[channels.index(c) for c in group]].mean(axis=1) for name,group in cfg['rois'].items()}
    values,schema={},{}
    def add(name,value,modality,family):
        values[name]=float(value);schema[name]={'modality':modality,'testing_family':family}
    for item in cfg['erp']:
        mask=window_mask(times,item['window']);wave=rois[item['roi']].mean(axis=0)[mask]
        if item.get('measure','amplitude')=='latency':
            peak=np.argmax(wave) if item['polarity']=='positive' else np.argmin(wave)
            value=times[mask][peak]*1000
        else:value=wave.mean()*1e6
        add(item['name'],value,'ERP','ERP')
    tf_cache={}
    for item in cfg['tf']:
        key=(tuple(item.get('all_frequencies',item['frequencies'])),tuple(item['baseline']))
        if key not in tf_cache:
            tf_cache[key]=power_db(epochs,sfreq,key[0],window_mask(times,key[1]))
        selected=np.isin(np.asarray(key[0]),item['frequencies'])
        ix=[channels.index(c) for c in cfg['rois'][item['roi']]]
        value=tf_cache[key][ix][:,selected][:,:,window_mask(times,item['window'])].mean()
        add(item['name'],value,'TF',item.get('testing_family',item['name']))
    if cfg['tf']:warnings.warn('TF uses reflected epoch padding; low-frequency edge estimates remain sensitive to short epochs.')
    signal_cache={}
    def signals(band):
        key=tuple(band)
        if key not in signal_cache:signal_cache[key]=analytic(epochs,band,sfreq)
        return signal_cache[key]
    for item in cfg['connectivity']:
        mask=window_mask(times,item['window']);z=signals(item['band'])[:,:,mask]
        estimates=[]
        for a in cfg['rois'][item['roi_a']]:
            for b in cfg['rois'][item['roi_b']]:
                cross=np.imag(z[:,channels.index(a)]*np.conj(z[:,channels.index(b)]))
                mode=cfg.get('dwpli_estimator','pooled_time')
                if mode=='trial_mean':cross=cross.mean(axis=1)
                elif mode!='pooled_time':raise ValueError('Unknown dwPLI estimator')
                estimates.append(dwpli_squared(cross))
        add(item['name'],np.mean(estimates),'dwPLI',item.get('testing_family',item['name']))
    if cfg['connectivity']:warnings.warn('dwPLI2 pools correlated within-trial time samples in pooled_time mode; effective sample size is lower. Retain finite negative estimates; use trial_mean as a separate sensitivity analysis.')
    rng=np.random.default_rng(cfg.get('seed',42))
    for item in cfg['pac']:
        mask=window_mask(times,item['window']);T=int(mask.sum())
        phase=np.angle(analytic(rois[item['phase_roi']],item['phase_band'],sfreq))[:,mask]
        amp=np.abs(analytic(rois[item['amplitude_roi']],item['amplitude_band'],sfreq))[:,mask]
        n_surrogates=item['surrogates']
        if n_surrogates<2:raise ValueError('At least two surrogates required')
        observed=np.mean([modulation_index(p,a) for p,a in zip(phase,amp)])
        null=[]
        lo=max(1,round(T*.1));hi=min(T,max(lo+1,round(T*.9)))
        for _ in range(n_surrogates):
            # Each surrogate repeats the same subject-level mean-MI statistic.
            shifts=rng.integers(lo,hi,size=len(phase))
            null.append(np.mean([modulation_index(np.roll(p,int(s)),a) for p,a,s in zip(phase,amp,shifts)]))
        sd=np.std(null,ddof=1)
        if not sd>1e-12:raise ValueError('PAC surrogate variance too small for zMI')
        add(item['name'],(observed-np.mean(null))/sd,'PAC',item.get('testing_family',item['name']))
    if cfg['pac']:warnings.warn('Short-window trial-wise MI can have sparse phase bins and strong finite-window bias; empty bins have zero mean amplitude. Whole-epoch filtering and time-shift surrogates do not establish valid low-frequency coupling. Inspect sensitivity before interpretation.')
    if not values or not np.isfinite(list(values.values())).all():raise ValueError('Nonfinite feature output')
    return values,schema
