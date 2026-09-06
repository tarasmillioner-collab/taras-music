"""Волны громкости: глубина провалов, дрожь. usage: python3 waves.py NAME file [NAME file ...]"""
import sys, numpy as np, librosa, scipy.signal as sg, json
sr=22050
def waves(f):
    y,_=librosa.load(f, sr=sr, mono=True); dur=len(y)/sr
    r=librosa.feature.rms(y=y, frame_length=2205, hop_length=1102)[0]; mdb=20*np.log10(r+1e-6); fs=sr/1102; t=np.arange(len(mdb))/fs
    b,a=sg.butter(2,[0.03/(fs/2),0.4/(fs/2)],btype='band'); slow=sg.filtfilt(b,a,mdb)
    core=(t>6)&(t<dur-6)
    tr,pr=sg.find_peaks(-slow, prominence=2.0, distance=int(4*fs))
    tr=[i for i in tr if core[i]]; depths=[float(-slow[i]) for i in tr]
    w=[np.mean(mdb[int(s*fs):int((s+3)*fs)]) for s in np.arange(6,dur-9,3)]; d=np.abs(np.diff(w))
    return dict(slow_std=round(float(slow[core].std()),2), drops=len(tr), drops_ge5=int(sum(x>=5 for x in depths)), drops_ge7=int(sum(x>=7 for x in depths)),
                max_drop=round(max(depths),1) if depths else 0, jitter_gt3=int((d>3).sum()), jitter_gt2=int((d>2).sum()),
                drop_times=[(round(t[i],1),round(-slow[i],1)) for i in tr if -slow[i]>=4])
res={}
for n,f in zip(sys.argv[1::2], sys.argv[2::2]):
    res[n]=waves(f); r=res[n]
    print(f"{n:8s} std {r['slow_std']:.2f}  провалов {r['drops']} (≥5dB {r['drops_ge5']}, ≥7dB {r['drops_ge7']}) max {r['max_drop']}  дрожь>3dB {r['jitter_gt3']} >2dB {r['jitter_gt2']}  | {r['drop_times']}")
json.dump(res,open("waves.json","w"),indent=1)
