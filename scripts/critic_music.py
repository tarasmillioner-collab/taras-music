"""КРИТИК №1 — МУЗЫКА v2. Откалиброван по оценкам заказчика: v2_E1≈7.5, v4_glued≈3.0.
usage: python3 critic_music.py NAME file [NAME file ...]
Два блока: A «похожесть и качество звука», B «энергия и жизнь». Итог = 10*(A^0.6 * B^0.4),
с потолком от худшей ключевой оси. Эталон сам себе обязан дать 10.0.
"""
import sys, os, re, json, subprocess
import numpy as np, librosa, scipy.signal as sg

SR = 22050
HERE = os.path.dirname(os.path.abspath(__file__))
REF_MIX = os.path.join(HERE, "full.wav")
# темп снимается с эталона, а не переносится между прогонами
def _ref_bpm(path):
    import librosa, numpy as np, librosa.feature.rhythm
    y, _ = librosa.load(path, sr=SR, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=SR)
    return float(np.atleast_1d(librosa.feature.rhythm.tempo(onset_envelope=env, sr=SR))[0])

BPM_REF = float(os.environ.get("BPM_REF") or (_ref_bpm(REF_MIX) if os.path.exists(REF_MIX) else 103.4))
BEAT_HZ = BPM_REF / 60

A_AXES = {"timbre": 2.0, "cohere": 1.8, "vocal_clean": 1.0, "spectrum": 0.9, "voice_bal": 1.1, "presence": 0.7}
B_AXES = {"intro_mode": 2.6, "intro_start": 1.6, "intro_energy": 1.2, "drive": 1.5, "waves": 1.2, "lifts": 1.0, "choir": 0.8, "tempo": 0.6, "form": 0.6}
KEY = ["timbre", "cohere", "intro_mode", "intro_start"]   # провал любой рушит итог


def db(x):
    return 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)


def feat(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    ys, _ = librosa.load(path, sr=SR, mono=False)
    if ys.ndim == 1:
        ys = np.vstack([ys, ys])
    dur = len(y) / SR
    f = {"dur": dur}
    yh, yp = librosa.effects.hpss(y)

    on = librosa.onset.onset_strength(y=yp, sr=SR)
    f["bpm"] = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=on, sr=SR))[0])

    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
    f["mfcc"] = m.mean(axis=1)

    S = np.abs(librosa.stft(y, n_fft=4096)) ** 2
    fr = librosa.fft_frequencies(sr=SR, n_fft=4096)
    tot = S.sum()
    f["bands"] = np.array([S[(fr >= a) & (fr < b)].sum() / tot * 100 for a, b in
                           [(20, 60), (60, 250), (250, 500), (500, 2000), (2000, 5000), (5000, 11025)]])

    # --- гармоническое к шумовому: франкенштейн и сепарация поднимают шум ---
    f["hnr"] = float(db(yh) - db(yp))

    # --- ВЧ: сепарация даёт фазовую кашу; меряем когерентность и стабильность спектра ---
    L, R = ys
    n = min(len(L), len(R))
    bl, al = sg.butter(4, [4000 / (SR / 2), 0.98], btype="band")
    Lh, Rh = sg.filtfilt(bl, al, L[:n]), sg.filtfilt(bl, al, R[:n])
    f["hf_coh"] = float(np.corrcoef(Lh, Rh)[0, 1])
    S2 = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    fr2 = librosa.fft_frequencies(sr=SR, n_fft=2048)
    hf = S2[(fr2 >= 4000) & (fr2 <= 9000)]
    hfe = hf.sum(0)
    act = hfe > np.percentile(hfe, 40)
    f["hf_wobble"] = float(np.std(np.diff(20 * np.log10(hfe[act] + 1e-9))))  # рваность ВЧ

    # --- ДРАЙВ: глубина модуляции огибающей на частоте доли и её гармониках ---
    r = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    fs = SR / 256
    e = r - r.mean()
    freqs, P = sg.welch(e, fs=fs, nperseg=2048)
    def peak_at(fq, tol=0.12):
        m_ = (freqs > fq * (1 - tol)) & (freqs < fq * (1 + tol))
        return float(P[m_].max()) if m_.any() else 0.0
    base = float(np.median(P[(freqs > 0.3) & (freqs < 8)]) + 1e-12)
    f["pulse"] = float(10 * np.log10((peak_at(BEAT_HZ) + peak_at(BEAT_HZ * 2) + peak_at(BEAT_HZ / 2)) / base))
    # ударность: доля транзиентов
    oenv = librosa.onset.onset_strength(y=y, sr=SR)
    f["onset_pk"] = float(np.percentile(oenv, 95) / (np.median(oenv) + 1e-9))

    # --- волны ---
    rr = librosa.feature.rms(y=y, frame_length=2205, hop_length=1102)[0]
    mdb = 20 * np.log10(rr + 1e-6)
    fs2 = SR / 1102
    t = np.arange(len(mdb)) / fs2
    bb, aa = sg.butter(2, [0.03 / (fs2 / 2), 0.4 / (fs2 / 2)], btype="band")
    slow = sg.filtfilt(bb, aa, mdb)
    core = (t > 6) & (t < dur - 6)
    tr, _ = sg.find_peaks(-slow, prominence=2.0, distance=int(4 * fs2))
    depth = [float(-slow[i]) for i in tr if core[i]]
    f["drops5"] = int(sum(d >= 5 for d in depth))
    f["max_drop"] = float(max(depth)) if depth else 0.0
    w = [np.mean(mdb[int(s * fs2):int((s + 3) * fs2)]) for s in np.arange(6, dur - 9, 3)]
    f["jitter3"] = int((np.abs(np.diff(w)) > 3).sum())

    # --- подбросы и хор по 10-с окнам ---
    ht = librosa.frames_to_time(np.arange(S2.shape[1]), sr=SR, hop_length=512)
    hi = S2[fr2 > 5000].sum(0)
    cen = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=512)[0]
    mid, side = (L[:n] + R[:n]) / 2, (L[:n] - R[:n]) / 2
    rows, widths = [], []
    for s in np.arange(0, dur - 5, 10):
        a, b = int(s * SR), int(min(dur, s + 10) * SR)
        m_ = (ht >= s) & (ht < s + 10)
        wdt = float(db(side[a:b]) - db(mid[a:b]))
        widths.append(wdt)
        rows.append([float(np.median(hi[m_])) if m_.any() else 0.0,
                     float(np.median(cen[m_])) if m_.any() else 0.0, wdt, float(db(y[a:b]))])
    rows = np.array(rows)
    z = (rows - rows.mean(0)) / (rows.std(0) + 1e-9)
    lift = np.clip(np.diff(z, axis=0), 0, None).sum(1)
    f["lifts"] = int((lift >= 2.0).sum())
    f["lift_max"] = float(lift.max()) if len(lift) else 0.0
    widths = np.array(widths)
    f["width_span"] = float(np.percentile(widths, 90) - np.percentile(widths, 10))  # хор появляется/уходит
    f["width_max"] = float(np.percentile(widths, 90))

    # --- СВЯЗНОСТЬ: живая запись держит контраст пик/впадина и связанные огибающие полос ---
    sc = librosa.feature.spectral_contrast(y=y, sr=SR, n_bands=6)
    f["contrast"] = sc.mean(axis=1)
    Sm = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=32, hop_length=512)
    E = 20 * np.log10(Sm + 1e-9)
    act2 = E.mean(0) > np.percentile(E.mean(0), 35)
    Ea = E[:, act2]
    C = np.corrcoef(Ea)
    iu = np.triu_indices(C.shape[0], k=3)
    f["band_link"] = float(np.nanmean(C[iu]))
    f["contrast_sd"] = float(sc.std(axis=1).mean())

    # --- ИНТРО: лад, старт, энергия первых секунд ---
    NAMES_ = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    MAJ_ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    MIN_ = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    intro = y[:int(15 * SR)]
    ih, _ = librosa.effects.hpss(intro)
    chi = librosa.feature.chroma_cqt(y=ih, sr=SR).mean(axis=1)
    cands = []
    for i in range(12):
        cands.append((np.corrcoef(chi, np.roll(MAJ_, i))[0, 1], i, 1))
        cands.append((np.corrcoef(chi, np.roll(MIN_, i))[0, 1], i, 0))
    cands.sort(reverse=True)
    _, root_, is_maj = cands[0]
    f["maj3_intro"] = float(chi[(root_ + 4) % 12] / (chi[(root_ + 3) % 12] + 1e-9))
    f["intro_is_major"] = int(is_maj)
    rr2 = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    tt2 = librosa.frames_to_time(np.arange(len(rr2)), sr=SR, hop_length=256)
    f["t_first_sound"] = float(tt2[np.argmax(rr2 > np.percentile(rr2, 60) * 0.35)])
    cen2 = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=512)[0]
    tc = librosa.frames_to_time(np.arange(len(cen2)), sr=SR, hop_length=512)
    f["bright_intro"] = float(np.median(cen2[tc < 15]))
    onall = librosa.onset.onset_detect(y=y, sr=SR, units="time", delta=0.08)
    f["events_intro"] = float(sum(1 for x in onall if x < 20) / 20)

    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    I = re.findall(r"I:\s+(-?[\d.]+) LUFS", out)
    LRA = re.findall(r"LRA:\s+([\d.]+) LU", out)
    f["lufs"] = float(I[-1]) if I else -99
    f["lra"] = float(LRA[-1]) if LRA else 0
    f["intro_db"] = float(db(y[:int(8 * SR)]) - db(y))
    f["ends_clean"] = float(mdb[-int(0.5 * fs2):].mean() - mdb.mean())
    ch = librosa.feature.chroma_cqt(y=yh, sr=SR).mean(axis=1)
    f["chroma"] = ch / (np.linalg.norm(ch) + 1e-9)
    return f


def rel(x, ref, tol):
    """1.0 при совпадении, 0 при отклонении tol (в тех же единицах)."""
    return float(np.clip(1 - abs(x - ref) / tol, 0, 1))


def score(f, R):
    s = {}
    d = np.linalg.norm(f["mfcc"] - R["mfcc"]) / (np.linalg.norm(R["mfcc"]) + 1e-9)
    s["timbre"] = float(np.clip(1 - d / 0.30, 0, 1))
    dc = np.linalg.norm(f["contrast"] - R["contrast"]) / (np.linalg.norm(R["contrast"]) + 1e-9)
    s["cohere"] = float(0.5 * np.clip(1 - dc / 0.16, 0, 1)
                        + 0.3 * np.clip(1 - abs(f["band_link"] - R["band_link"]) / 0.22, 0, 1)
                        + 0.2 * np.clip(1 - abs(f["contrast_sd"] - R["contrast_sd"]) / max(R["contrast_sd"] * 0.45, 1e-6), 0, 1))
    s["vocal_clean"] = float(0.5 * np.clip(1 - abs(f["hnr"] - R["hnr"]) / 3.0, 0, 1)
                             + 0.25 * np.clip(f["hf_coh"] / max(R["hf_coh"], 1e-6), 0, 1)
                             + 0.25 * np.clip(1 - abs(f["hf_wobble"] - R["hf_wobble"]) / max(R["hf_wobble"] * 0.5, 1e-6), 0, 1))
    relb = np.abs(f["bands"] - R["bands"]) / (R["bands"] + 1e-9)
    s["spectrum"] = float(np.clip(1 - relb.mean() / 0.45, 0, 1))
    s["voice_bal"] = float(0.6 * rel(f["bands"][3], R["bands"][3], R["bands"][3] * 0.40)
                           + 0.4 * rel(f["bands"][4], R["bands"][4], R["bands"][4] * 0.50))
    s["presence"] = float(rel(f["bands"][4] + f["bands"][5], R["bands"][4] + R["bands"][5],
                              (R["bands"][4] + R["bands"][5]) * 0.35))
    s["drive"] = float(0.6 * np.clip(f["pulse"] / max(R["pulse"], 1e-6), 0, 1.0)
                       + 0.4 * np.clip(f["onset_pk"] / max(R["onset_pk"], 1e-6), 0, 1.0))
    s["waves"] = float(0.4 * np.clip(1 - abs(f["jitter3"] - R["jitter3"]) / 8.0, 0, 1)
                       + 0.35 * np.clip(1 - abs(f["max_drop"] - R["max_drop"]) / 9.0, 0, 1)
                       + 0.25 * np.clip(1 - abs(f["drops5"] - R["drops5"]) / 5.0, 0, 1))
    s["lifts"] = float(0.6 * np.clip(1 - abs(f["lifts"] - R["lifts"]) / 4.0, 0, 1)
                       + 0.4 * np.clip(1 - abs(f["lift_max"] - R["lift_max"]) / 4.0, 0, 1))
    s["choir"] = float(0.6 * np.clip(1 - abs(f["width_span"] - R["width_span"]) / max(R["width_span"] * 0.6, 1e-6), 0, 1)
                       + 0.4 * np.clip(1 - abs(f["width_max"] - R["width_max"]) / 5.0, 0, 1))
    s["tempo"] = float(np.clip(1 - abs(f["bpm"] - R["bpm"]) / 6.0, 0, 1))
    # лад интро: мажор как у эталона обязателен
    s["intro_mode"] = float(0.7 * np.clip(f["maj3_intro"] / max(R["maj3_intro"], 1e-6), 0, 1)
                            + 0.3 * (1.0 if f["intro_is_major"] == R["intro_is_major"] else 0.0))
    s["intro_start"] = float(np.clip(1 - max(0, f["t_first_sound"] - R["t_first_sound"]) / 0.9, 0, 1))
    s["intro_energy"] = float(0.5 * np.clip(1 - abs(f["bright_intro"] - R["bright_intro"]) / (R["bright_intro"] * 0.28), 0, 1)
                              + 0.5 * np.clip(f["events_intro"] / max(R["events_intro"], 1e-6), 0, 1))
    s["form"] = float(0.4 * np.clip(1 - abs(f["intro_db"] - R["intro_db"]) / 4.0, 0, 1)
                      + 0.3 * np.clip(1 - abs(f["dur"] - R["dur"]) / 20.0, 0, 1)
                      + 0.3 * np.clip(1 - max(0, f["ends_clean"] - R["ends_clean"] - 3) / 12.0, 0, 1))

    def gmean(axes):
        num = sum(w * np.log(max(s[k], 0.02)) for k, w in axes.items())
        return float(np.exp(num / sum(axes.values())))
    A, B = gmean(A_AXES), gmean(B_AXES)
    total = 10 * (A ** 0.6) * (B ** 0.4)
    worst = min(s[k] for k in KEY)
    total = min(total, 10 * max(worst, 0.02) ** 0.42)   # потолок от худшей ключевой оси
    return s, A, B, float(total)


if __name__ == "__main__":
    R = feat(REF_MIX)
    keys = list(A_AXES) + list(B_AXES)
    print(f"{'трек':20s}" + "".join(f"{k[:6]:>8s}" for k in keys) + f"{'A':>6s}{'B':>6s}{'ИТОГ':>7s}")
    rs, ra, rb, rt = score(R, R)
    print(f"{'ЭТАЛОН self-test':20s}" + "".join(f"{rs[k]*10:8.1f}" for k in keys) + f"{ra*10:6.1f}{rb*10:6.1f}{rt:7.2f}")
    out = {}
    for n, p in zip(sys.argv[1::2], sys.argv[2::2]):
        f = feat(p)
        s, A, B, t = score(f, R)
        out[n] = {"axes": {k: round(s[k] * 10, 2) for k in keys}, "A": round(A * 10, 2), "B": round(B * 10, 2),
                  "total": round(t, 2), "raw": {k: round(float(v), 3) for k, v in f.items() if np.isscalar(v)}}
        print(f"{n:20s}" + "".join(f"{s[k]*10:8.1f}" for k in keys) + f"{A*10:6.1f}{B*10:6.1f}{t:7.2f}")
    p = os.path.join(HERE, "critic_music.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(out)
    json.dump(old, open(p, "w"), ensure_ascii=False, indent=1)
