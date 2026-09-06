"""РАЗРЕЖИВАНИЕ: у эталона в каждый момент звучит меньше слоёв, чем у клона.
Скрипт гасит лишние слои по карте эталона и ставит голос на его же уровень над музыкой.
usage: python3 thin.py STEMDIR OUT_TAG
"""
import sys, os, re, subprocess
import numpy as np, librosa, soundfile as sf, scipy.signal as sg

os.chdir(os.path.dirname(os.path.abspath(__file__)))
SR = 44100
src, tag = sys.argv[1], sys.argv[2]
REF = "../stems/htdemucs/full"
KEYS = ["vocals", "drums", "bass", "other"]

ref = {k: librosa.load(f"{REF}/{k}.wav", sr=SR, mono=True)[0] for k in KEYS}
st = {k: librosa.load(f"{src}/{k}.wav", sr=SR, mono=False)[0] for k in KEYS}
st = {k: (v if v.ndim == 2 else np.vstack([v, v])) for k, v in st.items()}
L = min(v.shape[1] for v in st.values())
st = {k: v[:, :L] for k, v in st.items()}

hop = 4410  # 100 мс


def lev(x, n):
    r = librosa.feature.rms(y=x, frame_length=hop * 2, hop_length=hop)[0][:n]
    return 20 * np.log10(r + 1e-6)


n = L // hop
lr = {k: lev(ref[k], n) for k in KEYS}
lc = {k: lev(st[k].mean(0), n) for k in KEYS}
m = min(min(len(x) for x in lr.values()), min(len(x) for x in lc.values()))
act_ref = sum((lr[k][:m] > (np.max(lr[k]) - 22)).astype(int) for k in KEYS)
act_cl = sum((lc[k][:m] > (np.max(lc[k]) - 22)).astype(int) for k in KEYS)
print(f"слоёв одновременно: эталон {act_ref.mean():.2f}, клон {act_cl.mean():.2f}")

# где у клона слоёв больше — душим самый тихий лишний слой (обычно other/bass)
order = ["other", "bass", "drums"]
gain = {k: np.zeros(m) for k in KEYS}
cut = 0
for i in range(m):
    extra = int(act_cl[i] - act_ref[i])
    if extra <= 0:
        continue
    live = [k for k in order if lc[k][i] > (np.max(lc[k]) - 22)]
    live.sort(key=lambda k: lc[k][i])          # сначала самый тихий
    for k in live[:extra]:
        gain[k][i] = -3.0
        cut += 1
print(f"кадров с лишними слоями: {cut} из {m * 3} возможных")

out = {}
for k in KEYS:
    g = sg.savgol_filter(gain[k], 9, 2) if m > 9 else gain[k]
    gl = np.repeat(10 ** (g / 20), hop)[:L]
    if len(gl) < L:
        gl = np.concatenate([gl, np.full(L - len(gl), gl[-1])])
    out[k] = st[k] * gl

# голос на уровень эталона в его полосе 300-3000 Гц
def band_ratio(v, mus):
    f = librosa.fft_frequencies(sr=SR, n_fft=4096)
    b = (f > 300) & (f < 3000)
    Sv = np.abs(librosa.stft(v, n_fft=4096)) ** 2
    Si = np.abs(librosa.stft(mus, n_fft=4096)) ** 2
    nn = min(Sv.shape[1], Si.shape[1])
    return 10 * np.log10(Sv[b][:, :nn].sum() / (Si[b][:, :nn].sum() + 1e-12))

tgt = band_ratio(ref["vocals"], ref["drums"] + ref["bass"] + ref["other"])
cur = band_ratio(out["vocals"].mean(0), (out["drums"] + out["bass"] + out["other"]).mean(0))
adj = float(np.clip((tgt - cur) * 0.45, -2.5, 2))
out["vocals"] = out["vocals"] * 10 ** (adj / 20)
print(f"голос над музыкой в полосе: эталон {tgt:.1f} dB, было {cur:.1f} → правка {adj:+.1f} dB")

mix = sum(out.values())
mix = mix / max(1e-9, np.abs(mix).max()) * 0.9
os.makedirs(f"stems/htdemucs/{tag}", exist_ok=True)
for k, v in out.items():
    sf.write(f"stems/htdemucs/{tag}/{k}.wav", v.T, SR)
sf.write(f"takes/{tag}.wav", mix.T, SR)
s = subprocess.run(["ffmpeg", "-hide_banner", "-i", f"takes/{tag}.wav", "-af", "ebur128", "-f", "null", "-"],
                   capture_output=True, text=True).stderr
I = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", s)[-1])
G = -14.2 - I
subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", f"takes/{tag}.wav",
                "-af", f"volume={G:.2f}dB,alimiter=limit=0.85:attack=5:release=80:level=false",
                "-b:a", "192k", f"takes/{tag}.mp3"])
print(f"готово: takes/{tag}.mp3")
