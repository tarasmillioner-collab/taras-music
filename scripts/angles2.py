"""УГЛЫ, КОТОРЫЕ ЕЩЁ НЕ СМОТРЕЛИ.
1) форманты вокала — почему голос звучит другим
2) динамика внутри фразы: нарастает или ровная
3) дыхание певицы: слышны ли вдохи и сколько
4) реверберация: время затухания
5) плотность аранжировки: сколько источников звучит одновременно
6) спектральный наклон — «тёплость» микса
7) маскировка вокала инструментами в его же полосе
usage: python3 angles2.py NAME stemdir mix.mp3 [...]
"""
import sys, os, json
import numpy as np, librosa, scipy.signal as sg

SR = 22050
HERE = os.path.dirname(os.path.abspath(__file__))


def db(x):
    return 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)


def look(stemdir, mixfile):
    st = {k: librosa.load(f"{stemdir}/{k}.wav", sr=SR, mono=True)[0]
          for k in ["vocals", "drums", "bass", "other"]}
    y, _ = librosa.load(mixfile, sr=SR, mono=True)
    v = st["vocals"]
    dur = len(v) / SR
    r = {}

    # --- 1. ФОРМАНТЫ вокала: пики огибающей спектра в 200-4000 Гц ---
    S = np.abs(librosa.stft(v, n_fft=2048))
    rms = librosa.feature.rms(y=v, frame_length=2048, hop_length=512)[0]
    act = rms > np.percentile(rms, 70)
    n = min(S.shape[1], len(act))
    avg = S[:, :n][:, act[:n]].mean(axis=1)
    fr = librosa.fft_frequencies(sr=SR, n_fft=2048)
    sm = sg.savgol_filter(20 * np.log10(avg + 1e-9), 21, 3)
    band = (fr > 200) & (fr < 4200)
    pk, _ = sg.find_peaks(sm[band], distance=8, prominence=1.5)
    fpk = fr[band][pk][:4]
    r["formants"] = [int(x) for x in fpk]

    # --- 2. ДИНАМИКА ВНУТРИ ФРАЗЫ: наклон громкости от начала к концу ---
    e = librosa.feature.rms(y=v, frame_length=1024, hop_length=256)[0]
    edb = 20 * np.log10(e + 1e-6)
    voiced = edb > edb.max() - 28
    runs, start = [], None
    for i, x in enumerate(voiced):
        if x and start is None:
            start = i
        elif not x and start is not None:
            if (i - start) * 256 / SR > 0.6:
                runs.append((start, i))
            start = None
    slopes = []
    for a, b in runs:
        seg = edb[a:b]
        if len(seg) > 8:
            slopes.append(np.polyfit(np.arange(len(seg)), seg, 1)[0] * len(seg))
    r["phrase_shape_db"] = round(float(np.median(slopes)), 2) if slopes else None
    r["phrase_rising_pct"] = round(float(np.mean(np.array(slopes) > 1) * 100), 1) if slopes else None

    # --- 3. ДЫХАНИЕ: короткие шумовые всплески между фразами ---
    hi = sg.sosfilt(sg.butter(4, [3000 / (SR / 2), 0.98], btype="band", output="sos"), v)
    he = librosa.feature.rms(y=hi, frame_length=1024, hop_length=256)[0]
    zc = librosa.feature.zero_crossing_rate(v, frame_length=1024, hop_length=256)[0]
    gaps = ~voiced
    breath = (he > np.percentile(he, 60)) & (zc > np.percentile(zc, 70)) & gaps[:len(he)]
    r["breaths_per_min"] = round(float(breath.sum() * 256 / SR / (dur / 60)), 1)

    # --- 4. РЕВЕРБЕРАЦИЯ: спад энергии после окончаний фраз ---
    tails = []
    for a, b in runs:
        s0, s1 = b, min(b + int(0.7 * SR / 256), len(edb) - 1)
        if s1 - s0 > 6:
            seg = edb[s0:s1]
            k = np.polyfit(np.arange(len(seg)), seg, 1)[0]
            if k < 0:
                tails.append(-60.0 / (k / (256 / SR)))
    r["rt60_ms"] = round(float(np.median(tails) * 1000), 0) if tails else None

    # --- 5. ПЛОТНОСТЬ: сколько стемов звучит одновременно ---
    lev = {}
    hop = 4410
    for k, x in st.items():
        rr = librosa.feature.rms(y=x, frame_length=hop * 2, hop_length=hop)[0]
        lev[k] = 20 * np.log10(rr + 1e-6)
    m = min(len(x) for x in lev.values())
    act_cnt = sum((lev[k][:m] > (np.max(lev[k]) - 22)).astype(int) for k in lev)
    r["layers_mean"] = round(float(act_cnt.mean()), 2)
    r["layers_sd"] = round(float(act_cnt.std()), 2)

    # --- 6. СПЕКТРАЛЬНЫЙ НАКЛОН микса ---
    Sm = np.abs(librosa.stft(y, n_fft=4096)) ** 2
    frm = librosa.fft_frequencies(sr=SR, n_fft=4096)
    b_ = (frm > 100) & (frm < 10000)
    p = np.polyfit(np.log10(frm[b_]), 10 * np.log10(Sm[b_].mean(axis=1) + 1e-12), 1)
    r["spectral_tilt_db_dec"] = round(float(p[0]), 1)

    # --- 7. МАСКИРОВКА: вокал против остального в полосе 300-3000 ---
    bmask = (frm > 300) & (frm < 3000)
    Sv = np.abs(librosa.stft(v, n_fft=4096)) ** 2
    Si = np.abs(librosa.stft(st["drums"] + st["bass"] + st["other"], n_fft=4096)) ** 2
    nn = min(Sv.shape[1], Si.shape[1])
    r["voice_over_music_in_band_db"] = round(float(
        10 * np.log10(Sv[bmask][:, :nn].sum() / (Si[bmask][:, :nn].sum() + 1e-12))), 1)
    return r


if __name__ == "__main__":
    a = sys.argv[1:]
    out = {}
    for i in range(0, len(a), 3):
        out[a[i]] = look(a[i + 1], a[i + 2])
    keys = list(next(iter(out.values())).keys())
    print(f"{'угол':26s}" + "".join(f"{n:>14s}" for n in out))
    for k in keys:
        print(f"{k:26s}" + "".join(f"{str(out[n][k]):>14s}" for n in out))
    json.dump(out, open(os.path.join(HERE, "angles2.json"), "w"), ensure_ascii=False, indent=1)
