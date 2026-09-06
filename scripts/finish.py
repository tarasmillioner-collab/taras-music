"""ФИНИШ: единый пост-процесс дубля под эталон.
1) matching EQ по 48 полосам  2) прицельная спектральная экспансия 3-9 кГц
3) перенос медленной огибающей эталона (волны и провалы)  4) мастер -14.2 LUFS + лимитер
usage: python3 finish.py IN.mp3 OUT_TAG [--eq 1] [--xp 1.2] [--wv 1.0] [--lines FILE]
"""
import sys, os, re, subprocess, json
import numpy as np, librosa, soundfile as sf, scipy.signal as sg

os.chdir(os.path.dirname(os.path.abspath(__file__)))
SR = 44100
inp, tag = sys.argv[1], sys.argv[2]
A = sys.argv[3:]
def opt(name, dflt):
    return float(A[A.index(name) + 1]) if name in A else dflt
EQ, XP, WV = opt("--eq", 1.0), opt("--xp", 1.2), opt("--wv", 1.0)
LINES = A[A.index("--lines") + 1] if "--lines" in A else None

_REF = next((q for q in (os.environ.get("REF_WAV"), "full.wav", "../full.wav") if q and os.path.exists(q)), None)
if _REF is None:
    sys.exit("не найден эталон: положи full.wav рядом со скриптом или задай REF_WAV=/путь/full.wav")
ref, _ = librosa.load(_REF, sr=SR, mono=True)
y, _ = librosa.load(inp, sr=SR, mono=False)
if y.ndim == 1:
    y = np.vstack([y, y])
L = y.shape[1]

# ---------- 0. срез тишины в начале: эталон стартует на 0.07 с ----------
r0 = librosa.feature.rms(y=y.mean(0), frame_length=1024, hop_length=256)[0]
t0 = librosa.frames_to_time(np.arange(len(r0)), sr=SR, hop_length=256)
onset_t = float(t0[np.argmax(r0 > np.percentile(r0, 60) * 0.35)])
cut = max(0.0, onset_t - 0.07)
if cut > 0.05:
    y = y[:, int(cut * SR):]
    L = y.shape[1]
    print(f"0) срез тишины: -{cut:.2f} с, старт теперь на 0.07 с")

# ---------- 1. matching EQ ----------
def avg_spec(x, nfft=8192):
    S = np.abs(librosa.stft(x, n_fft=nfft, hop_length=nfft // 4))
    r = librosa.feature.rms(y=x, frame_length=nfft, hop_length=nfft // 4)[0]
    act = r > np.percentile(r, 35)
    n = min(S.shape[1], len(act))
    return S[:, :n][:, act[:n]].mean(axis=1), librosa.fft_frequencies(sr=SR, n_fft=nfft)

Sr, fr = avg_spec(ref)
edges = np.geomspace(30, 18000, 49)

def match_eq(x, amount=1.0, label="EQ"):
    Sy, _ = avg_spec(x.mean(0))
    g, c = [], []
    for i in range(48):
        m = (fr >= edges[i]) & (fr < edges[i + 1])
        if not m.any():
            continue
        g.append(np.clip(20 * np.log10((Sr[m].mean() + 1e-12) / (Sy[m].mean() + 1e-12)), -9, 9))
        c.append(np.sqrt(edges[i] * edges[i + 1]))
    g = sg.savgol_filter(np.array(g) * amount, 7, 2)
    g -= np.median(g)
    fp = np.concatenate([[0], np.array(c), [SR / 2]]) / (SR / 2)
    fp[0], fp[-1] = 0, 1
    fp = np.maximum.accumulate(fp)
    t = sg.firwin2(2049, fp, 10 ** (np.concatenate([[g[0]], g, [g[-1]]]) / 20))
    print(f"{label}: {g.min():+.1f}…{g.max():+.1f} dB")
    return np.stack([sg.fftconvolve(x[ch], t, mode="same")[:L] for ch in range(2)])

y = match_eq(y, EQ, "1) EQ")

# ---------- 2. спектральная экспансия, прицельно 3-9 кГц ----------
if XP > 0:
    nfft, hop = 2048, 512
    fb = librosa.fft_frequencies(sr=SR, n_fft=nfft)
    wf = np.clip((fb - 350) / 350, 0, 1) * np.clip((14000 - fb) / 3000, 0, 1)
    # опора — медиана внутри ОКТАВНОЙ полосы (так же, как её видит spectral_contrast)
    oct_edges = [200, 400, 800, 1600, 3200, 6400, 12800, SR / 2]
    masks = [((fb >= a) & (fb < b)) for a, b in zip(oct_edges[:-1], oct_edges[1:])]
    out = []
    for ch in range(2):
        Z = librosa.stft(y[ch], n_fft=nfft, hop_length=hop)
        mag, ph = np.abs(Z), np.angle(Z)
        mdb = 20 * np.log10(mag + 1e-9)
        loc = np.zeros_like(mdb)
        for mk in masks:
            if mk.sum() < 4:
                continue
            med = np.median(mdb[mk], axis=0, keepdims=True)      # медиана по бинам полосы в каждом кадре
            loc[mk] = med
        below = np.clip(loc - mdb, 0, 40)
        gdb = np.clip(-XP * below * 1.0 * wf[:, None], -30, 0)
        gdb = sg.convolve2d(gdb, np.ones((3, 3)) / 9.0, mode="same", boundary="symm")
        out.append(librosa.istft(mag * 10 ** (gdb / 20) * np.exp(1j * ph), hop_length=hop, length=L))
    y = np.stack(out)
    print(f"2) экспансия, сила {XP}")
    y = match_eq(y, 1.0, "2b) EQ после экспансии")

# ---------- 3. перенос волн эталона ----------
if WV > 0:
    hop = 441
    def env(x):
        r = librosa.feature.rms(y=x, frame_length=4410, hop_length=hop)[0]
        e = 20 * np.log10(r + 1e-6)
        k = 60
        return np.convolve(np.pad(e, k, mode="edge"), np.ones(k) / k, mode="same")[k:-k]
    er, ey = env(ref), env(y.mean(0))
    n = min(len(er), len(ey))
    if LINES and os.path.exists(LINES):
        ref_l = json.load(open("../lyrics.json"))
        m_l = json.load(open(LINES))
        idx = list(range(0, 8)) + [8, 8] + list(range(9, 74))
        pairs = [(ref_l[idx[i]][0], l["t"]) for i, l in enumerate(m_l) if l["t"] is not None and i < len(idx)]
        rt = np.maximum.accumulate(np.array([0.0] + [a for a, _ in pairs] + [L / SR]))
        mt = np.maximum.accumulate(np.array([0.0] + [b for _, b in pairs] + [L / SR]))
        tm = np.arange(n) * hop / SR
        er = np.interp(np.interp(tm, mt, rt), np.arange(len(er)) * hop / SR, er)
        print(f"3) волны: варп по {len(pairs)} строкам")
    else:
        cc = sg.correlate(ey[:n] - ey[:n].mean(), er[:n] - er[:n].mean(), mode="full")
        lags = np.arange(-n + 1, n)
        w = np.abs(lags) <= int(2.0 * SR / hop)
        lag = int(lags[np.argmax(np.where(w, cc, -np.inf))])
        er = np.roll(er, lag)
        print(f"3) волны: сдвиг {lag*hop/SR:+.2f} с")
    er = er - np.median(er[:n])
    ey2 = ey[:n] - np.median(ey[:n])
    gain = np.clip((er[:n] - ey2) * WV, -7, 5)
    gl = np.repeat(10 ** (gain / 20), hop)
    gl = np.concatenate([gl, np.full(max(0, L - len(gl)), gl[-1])])[:L]
    y = y * gl
    print(f"   гейн волн: медиана {np.median(gain):+.1f}, p5 {np.percentile(gain,5):+.1f}, p95 {np.percentile(gain,95):+.1f} dB")

# ---------- 4. мастер ----------
y = y / max(1e-9, np.abs(y).max()) * 0.9
sf.write(f"takes/{tag}.wav", y.T, SR)
s = subprocess.run(["ffmpeg", "-hide_banner", "-i", f"takes/{tag}.wav", "-af", "ebur128", "-f", "null", "-"],
                   capture_output=True, text=True).stderr
I = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", s)[-1])
G = -14.2 - I
subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", f"takes/{tag}.wav",
                "-af", f"volume={G:.2f}dB,alimiter=limit=0.85:attack=5:release=80:level=false",
                "-b:a", "192k", f"takes/{tag}.mp3"])
print(f"4) мастер {G:+.1f} dB → takes/{tag}.mp3")
