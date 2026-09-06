"""RETIME: ставит каждую спетую строку клона на таймкод эталона (строка в строку).
usage: python3 retime.py IN.mp3 LINES.json OUT_TAG
Между якорями фрагмент растягивается или сжимается фазовым вокодером; коэффициенты
печатаются, чтобы видеть, где растяжение сильное.
"""
import sys, os, re, json, subprocess
import numpy as np, librosa, soundfile as sf

os.chdir(os.path.dirname(os.path.abspath(__file__)))
SR = 44100
inp, lines_json, tag = sys.argv[1], sys.argv[2], sys.argv[3]
ref = json.load(open(os.environ.get("REF","../lyrics.json")))
lines = json.load(open(lines_json))
IDX = list(range(len(ref))) if os.environ.get("REF") else list(range(0, 8)) + [8, 8] + list(range(9, 74))

y, _ = librosa.load(inp, sr=SR, mono=False)
if y.ndim == 1:
    y = np.vstack([y, y])
L = y.shape[1]
dur = L / SR

# якоря: (время в клоне, время в эталоне)
pairs = []
for i, l in enumerate(lines):
    if i >= len(IDX) or l["t"] is None:
        continue
    rt = ref[IDX[i]][0]
    if pairs and (l["t"] <= pairs[-1][0] + 0.25 or rt <= pairs[-1][1] + 0.25):
        continue                                    # монотонность
    pairs.append((float(l["t"]), float(rt)))
pairs = [(0.0, 0.0)] + pairs + [(dur, dur)]
print(f"якорей: {len(pairs)-2}")

# --- сглаживание: гасим накопленный дрейф плавно, а не по каждой строке ---
import scipy.signal as _sg
u = np.array([a for a, _ in pairs])
r = np.array([b for _, b in pairs])
d = r - u                                   # дрейф: сколько клон опережает эталон
k = int(os.environ.get("SMOOTH", 7))
d_s = np.convolve(np.pad(d, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
d_s[0] = 0.0
d_s[-1] = d[-1]
r_s = u + d_s
r_s = np.maximum.accumulate(r_s)            # монотонность обязательна
rate = np.diff(u) / np.maximum(np.diff(r_s), 1e-6)
bad = (rate < 0.85) | (rate > 1.18)
if bad.any():                               # мягко подрезаем выбросы и пересобираем
    rate = np.clip(rate, 0.85, 1.18)
    r_s = np.concatenate([[0.0], np.cumsum(np.diff(u) / rate)])
pairs = list(zip(u, r_s))
resid = np.abs(r_s - r)
print(f"дрейф до: медиана {np.median(np.abs(d)):.2f} с, макс {np.abs(d).max():.2f} с")
print(f"дрейф после: медиана {np.median(resid):.2f} с, макс {resid.max():.2f} с")

segs = []
ratios = []
for k in range(len(pairs) - 1):
    a0, b0 = pairs[k]
    a1, b1 = pairs[k + 1]
    src = y[:, int(a0 * SR):int(a1 * SR)]
    if src.shape[1] < 512:
        continue
    want = max(1, int((b1 - b0) * SR))
    rate = src.shape[1] / want                       # >1 = сжать, <1 = растянуть
    ratios.append(rate)
    if abs(rate - 1) < 0.008:
        out = src
    else:
        out = np.stack([librosa.effects.time_stretch(src[c], rate=rate) for c in range(2)])
    # подгонка длины до нужной ровно
    if out.shape[1] > want:
        out = out[:, :want]
    elif out.shape[1] < want:
        out = np.pad(out, ((0, 0), (0, want - out.shape[1])), mode="edge")
    segs.append(out)

# склейка с короткими кроссфейдами, чтобы не щёлкало
xf = int(0.012 * SR)
res = segs[0]
for s in segs[1:]:
    n = min(xf, res.shape[1], s.shape[1])
    if n > 8:
        w = np.linspace(0, 1, n)
        tail = res[:, -n:] * (1 - w) + s[:, :n] * w
        res = np.concatenate([res[:, :-n], tail, s[:, n:]], axis=1)
    else:
        res = np.concatenate([res, s], axis=1)

r = np.array(ratios)
print(f"коэффициенты: медиана {np.median(r):.3f}, мин {r.min():.3f}, макс {r.max():.3f}, "
      f"сильнее 15 %: {(np.abs(r-1) > 0.15).sum()} из {len(r)}")
res = res / max(1e-9, np.abs(res).max()) * 0.9
sf.write(f"takes/{tag}.wav", res.T, SR)
s = subprocess.run(["ffmpeg", "-hide_banner", "-i", f"takes/{tag}.wav", "-af", "ebur128", "-f", "null", "-"],
                   capture_output=True, text=True).stderr
I = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", s)[-1])
G = -14.2 - I
subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", f"takes/{tag}.wav",
                "-af", f"volume={G:.2f}dB,alimiter=limit=0.85:attack=5:release=80:level=false",
                "-b:a", "192k", f"takes/{tag}.mp3"])
print(f"готово: takes/{tag}.mp3, длительность {res.shape[1]/SR:.1f} с (эталон 194.4)")
