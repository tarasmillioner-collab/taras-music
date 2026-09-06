import sys, json, numpy as np, librosa, scipy.signal as sg
sr=22050; BPM=103.4; beat=60/BPM; bar=4*beat
def db(x): return 20*np.log10(np.sqrt((x**2).mean())+1e-9)
def load(d): return {k: librosa.load(f"{d}/{k}.wav", sr=sr, mono=True)[0] for k in ["vocals","drums","bass","other"]}
def analyze(name, d):
    st=load(d); v=st["vocals"]; mix=sum(st.values()); dur=len(v)/sr; R={}
    # ---- vocal pitch contour ----
    f0,vf,_=librosa.pyin(v, fmin=librosa.note_to_hz('E3'), fmax=librosa.note_to_hz('C6'), sr=sr, frame_length=2048, hop_length=256)
    t=librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=256)
    midi=librosa.hz_to_midi(f0); voiced=~np.isnan(midi)
    # phrases = voiced runs separated by >0.25s unvoiced
    runs=[]; start=None
    for i in range(len(voiced)):
        if voiced[i] and start is None: start=i
        if (not voiced[i] or i==len(voiced)-1) and start is not None:
            if start is not None and (i-start)*256/sr>0.35: runs.append((start,i))
            start=None
    # merge runs separated by <0.25s
    merged=[]
    for a,b in runs:
        if merged and (a-merged[-1][1])*256/sr<0.25: merged[-1]=(merged[-1][0],b)
        else: merged.append((a,b))
    ranges=[]; rises=0; falls=0; leaps=0; steps=0; move=[]
    for a,b in merged:
        m=midi[a:b]; m=m[~np.isnan(m)]
        if len(m)<8: continue
        ranges.append(np.percentile(m,95)-np.percentile(m,5))
        head=np.median(m[:len(m)//4]); tail=np.median(m[-len(m)//4:])
        if tail-head>1: rises+=1
        elif head-tail>1: falls+=1
        dm=np.abs(np.diff(sg.medfilt(m,9)))
        leaps+=int((dm>=2.5).sum()); steps+=int(((dm>=0.7)&(dm<2.5)).sum())
        move.append(np.abs(np.diff(sg.medfilt(m,9))).sum()/((b-a)*256/sr))
    mv=midi[voiced]
    R["phrases"]=len(merged); R["phrase_range_st_median"]=round(float(np.median(ranges)),2); R["phrase_range_st_p75"]=round(float(np.percentile(ranges,75)),2)
    R["phrase_end_rising_pct"]=round(rises/max(1,len(ranges))*100); R["phrase_end_falling_pct"]=round(falls/max(1,len(ranges))*100)
    R["leaps_per_min"]=round(leaps/(dur/60),1); R["steps_per_min"]=round(steps/(dur/60),1)
    R["pitch_motion_st_per_s"]=round(float(np.median(move)),2)
    R["pitch_std_st"]=round(float(np.std(mv)),2); R["pitch_range_p5_p95"]=round(float(np.percentile(mv,95)-np.percentile(mv,5)),1)
    R["pct_time_top_quartile"]=round(float((mv>np.percentile(mv,75)).mean()*100),1)
    # vibrato: 4-8 Hz modulation on sustained frames
    dev=midi-sg.medfilt(np.nan_to_num(midi, nan=np.nanmedian(midi)),31)
    R["vibrato_depth_cents"]=round(float(np.nanstd(dev[voiced])*100),1)
    # ---- vocal micro-dynamics ----
    vr=librosa.feature.rms(y=v, frame_length=2205, hop_length=1102)[0]; vdb=20*np.log10(vr+1e-6); vt=librosa.frames_to_time(np.arange(len(vr)),sr=sr,hop_length=1102)
    on=vdb>vdb.max()-35; R["vocal_dyn_std_db"]=round(float(vdb[on].std()),2)
    # swells: local maxima exceeding the min within ±1.5s by >=5 dB
    pk,_=sg.find_peaks(vdb, prominence=5, distance=10); R["vocal_swells_per_min"]=round(len(pk)/(dur/60),1)
    cen=librosa.feature.spectral_centroid(y=v, sr=sr, hop_length=1102)[0]; R["vocal_brightness_std_hz"]=round(float(cen[on].std()))
    # harmonies / width
    # ---- mix slow waves ----
    mr=librosa.feature.rms(y=mix, frame_length=2205, hop_length=1102)[0]; mdb=20*np.log10(mr+1e-6); fs=sr/1102
    b,a=sg.butter(2,[0.03/(fs/2),0.4/(fs/2)],btype='band'); slow=sg.filtfilt(b,a,mdb)
    R["mix_slow_wave_std_db"]=round(float(slow[int(3*fs):-int(3*fs)].std()),2)
    b,a=sg.butter(2,[0.5/(fs/2),4/(fs/2)],btype='band'); fast=sg.filtfilt(b,a,mdb); R["mix_pump_std_db"]=round(float(fast.std()),2)
    # section contrast: 8-bar blocks RMS spread
    blk=8*bar; sec=[db(mix[int(s*sr):int((s+blk)*sr)]) for s in np.arange(0,dur-blk,blk)]; R["section_contrast_db_std"]=round(float(np.std(sec)),2); R["section_max_minus_min_db"]=round(float(max(sec)-min(sec)),2)
    # ---- novelty events ----
    S=np.abs(librosa.stft(mix, n_fft=2048, hop_length=512)); flux=np.maximum(0,np.diff(S,axis=1)).sum(0); flux=flux/np.percentile(flux,99)
    nov=librosa.util.normalize(flux); ft=librosa.frames_to_time(np.arange(len(nov)),sr=sr,hop_length=512)
    # novelty smoothed at 1s, peaks with prominence
    novs=sg.savgol_filter(nov, 43, 3); pk,_=sg.find_peaks(novs, prominence=0.12, distance=int(1.0*sr/512)); R["mix_novelty_events_per_min"]=round(len(pk)/(dur/60),1)
    for k in ["drums","other"]:
        o=librosa.onset.onset_strength(y=st[k], sr=sr); os_=sg.savgol_filter(o,21,3); pk,_=sg.find_peaks(os_, prominence=np.percentile(os_,90)-np.percentile(os_,50), distance=int(0.5*sr/512))
        R[f"{k}_accents_per_min"]=round(len(pk)/(dur/60),1)
    # drum groove variation per bar (hi-hat band density)
    Sd=np.abs(librosa.stft(st["drums"], n_fft=1024, hop_length=256)); fr=librosa.fft_frequencies(sr=sr,n_fft=1024); hi=Sd[fr>5000].sum(0); low=Sd[(fr>40)&(fr<150)].sum(0)
    tt=librosa.frames_to_time(np.arange(len(hi)),sr=sr,hop_length=256)
    def per_bar(x):
        out=[]
        for s in np.arange(0,dur-bar,bar):
            m=(tt>=s)&(tt<s+bar); seg=x[m]; thr=np.percentile(x,80); out.append(int(((seg>thr)&(np.r_[True,seg[1:]>seg[:-1]])).sum()))
        return np.array(out)
    hb=per_bar(hi); kb=per_bar(low)
    act=hb[hb>0]; R["hat_hits_per_bar_median"]=int(np.median(act)) if len(act) else 0; R["hat_hits_per_bar_std"]=round(float(act.std()),2) if len(act) else 0
    R["kick_hits_per_bar_std"]=round(float(kb[kb>0].std()),2) if (kb>0).any() else 0
    # chord change rate: chroma distance bar to bar
    ch=librosa.feature.chroma_cqt(y=st["other"], sr=sr); cb=[]
    for s in np.arange(0.12,dur-bar,bar):
        f=librosa.time_to_frames([s,s+bar],sr=sr); c=ch[:,f[0]:f[1]].mean(1); cb.append(c/(np.linalg.norm(c)+1e-9))
    cb=np.array(cb); dist=1-np.sum(cb[1:]*cb[:-1],axis=1); R["chord_change_pct_bars"]=round(float((dist>0.15).mean()*100))
    # vocal vs music per 8 bars (how much the vocal 'rides' above the music)
    vm=[db(v[int(s*sr):int((s+blk)*sr)])-db((st["drums"]+st["bass"]+st["other"])[int(s*sr):int((s+blk)*sr)]) for s in np.arange(0,dur-blk,blk)]
    R["vocal_over_music_std_db"]=round(float(np.std(vm)),2)
    R["drums_over_other_db"]=round(float(db(st["drums"])-db(st["other"])),1)
    R["bass_over_other_db"]=round(float(db(st["bass"])-db(st["other"])),1)
    return R
names=sys.argv[1::2]; dirs=sys.argv[2::2]
res={n:analyze(n,d) for n,d in zip(names,dirs)}
json.dump(res,open("micro_dna.json","w"),indent=1)
keys=list(res[names[0]].keys())
print(f"{'метрика':32s}"+"".join(f"{n:>12s}" for n in names))
for k in keys: print(f"{k:32s}"+"".join(f"{str(res[n][k]):>12s}" for n in names))
