"""Подбросы: таймлайн по 10с окнам + индекс подъёма. usage: python3 emotion_dna.py NAME stemdir mixfile [...]"""
import sys, json, numpy as np, librosa, scipy.signal as sg
sr=22050; W=10.0
def db(x): return 20*np.log10(np.sqrt((x**2).mean())+1e-9)
def timeline(d, mixf):
    st={k: librosa.load(f"{d}/{k}.wav", sr=sr, mono=False)[0] for k in ["vocals","drums","bass","other"]}
    mono={k:v.mean(0) for k,v in st.items()}; v=mono["vocals"]; dur=len(v)/sr
    f0,_,_=librosa.pyin(v, fmin=librosa.note_to_hz('E3'), fmax=librosa.note_to_hz('C6'), sr=sr, frame_length=2048, hop_length=256)
    ft=librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=256); midi=librosa.hz_to_midi(f0)
    cen=librosa.feature.spectral_centroid(y=v, sr=sr, hop_length=256)[0]
    Sd=np.abs(librosa.stft(mono["drums"], n_fft=1024, hop_length=256)); fr=librosa.fft_frequencies(sr=sr,n_fft=1024); hi=Sd[fr>5000].sum(0); ht=librosa.frames_to_time(np.arange(len(hi)),sr=sr,hop_length=256); thr=np.percentile(hi,80)
    mix=sum(st.values()); Sm=np.abs(librosa.stft(mix.mean(0), n_fft=4096, hop_length=1024))**2; frm=librosa.fft_frequencies(sr=sr,n_fft=4096); mt=librosa.frames_to_time(np.arange(Sm.shape[1]),sr=sr,hop_length=1024)
    rows=[]
    for s in np.arange(0,dur-W/2,W):
        a,b=int(s*sr),int(min(dur,s+W)*sr); m=(ft>=s)&(ft<s+W)&(~np.isnan(midi))
        vs=st["vocals"][:,a:b]; ms=mix[:,a:b]
        row={"t":float(s)}
        row["p90"]=round(float(np.percentile(midi[m],90)),1) if m.sum()>20 else None
        row["p50"]=round(float(np.percentile(midi[m],50)),1) if m.sum()>20 else None
        row["voc_bright"]=round(float(np.median(cen[m]))) if m.sum()>20 else None
        row["voc_width"]=round(float(db((vs[0]-vs[1])/2)-db((vs[0]+vs[1])/2)),1)
        row["mix_width"]=round(float(db((ms[0]-ms[1])/2)-db((ms[0]+ms[1])/2)),1)
        row["voc_lvl"]=round(float(db(v[a:b])),1)
        music=(mono["drums"]+mono["bass"]+mono["other"])[a:b]; row["voc_over_music"]=round(float(db(v[a:b])-db(music)),1)
        hm=(ht>=s)&(ht<s+W); seg=hi[hm]; row["hat"]=round(float(((seg>thr)&(np.r_[True,seg[1:]>seg[:-1]])).sum()/(W/(4*60/103.4))),1)
        mm=(mt>=s)&(mt<s+W); tot=Sm[:,mm].sum(); row["sub_pct"]=round(float(Sm[(frm>=20)&(frm<80)][:,mm].sum()/tot*100),1); row["air_pct"]=round(float(Sm[frm>=5000][:,mm].sum()/tot*100),1)
        row["bass_lvl"]=round(float(db(mono["bass"][a:b])),1); row["drums_lvl"]=round(float(db(mono["drums"][a:b])),1); row["mix_lvl"]=round(float(db(mix.mean(0)[a:b])),1)
        rows.append(row)
    # lift index: sum of standardized positive jumps in p90, voc_width, hat, mix_lvl, air between consecutive windows
    keys=["p90","voc_width","hat","mix_lvl","air_pct"]; lifts=[]
    for i in range(1,len(rows)):
        sc=0; parts=[]
        for k in keys:
            a_,b_=rows[i-1][k],rows[i][k]
            if a_ is None or b_ is None: continue
            vals=[r[k] for r in rows if r[k] is not None]; sd=np.std(vals)+1e-6; z=(b_-a_)/sd
            if z>0.5: sc+=z; parts.append(f"{k}+{z:.1f}")
        lifts.append((rows[i]["t"],round(sc,1),parts))
    return rows,lifts
names=sys.argv[1::3]; out={}
for n,d,mf in zip(sys.argv[1::3],sys.argv[2::3],sys.argv[3::3]):
    rows,lifts=timeline(d,mf); out[n]={"rows":rows,"lifts":lifts}
    print(f"===== {n}")
    print(" t    p90  p50 bright vocW mixW vocLvl v/m  hat  sub  air  bass drums mix")
    for r in rows: print(f"{r['t']:4.0f} {str(r['p90']):>5} {str(r['p50']):>5} {str(r['voc_bright']):>6} {r['voc_width']:5} {r['mix_width']:5} {r['voc_lvl']:6} {r['voc_over_music']:4} {r['hat']:4} {r['sub_pct']:4} {r['air_pct']:4} {r['bass_lvl']:5} {r['drums_lvl']:5} {r['mix_lvl']:5}")
    big=[l for l in lifts if l[1]>=2.0]; print(f" подбросов (индекс≥2): {len(big)} →", [(t,s) for t,s,_ in big]); print(" состав:", [(t,p) for t,s,p in big])
json.dump(out,open("emotion_dna.json","w"),indent=0)
