"""ПОЧИНКА МИКСА ПО СТЕМАМ — по разбору слухового критика:
вокал (де-эссер, HPF, приглушение хвоста реверба), ударные (транзиент, панч кика),
бас (HPF + дакинг от кика), гармония (вырез 200-400 Гц + расширение стерео).
usage: python3 mixfix.py TAKE_STEMDIR OUT_TAG
"""
import sys, os, re, subprocess
import numpy as np, librosa, soundfile as sf, scipy.signal as sg

os.chdir(os.path.dirname(os.path.abspath(__file__)))
SR = 44100
src, tag = sys.argv[1], sys.argv[2]
st = {k: librosa.load(f"{src}/{k}.wav", sr=SR, mono=False)[0] for k in ["vocals", "drums", "bass", "other"]}
L = min(v.shape[1] for v in st.values())
st = {k: v[:, :L] for k, v in st.items()}


def bq(x, kind, f0, gain_db=0.0, q=0.9):
    w0 = 2 * np.pi * f0 / SR
    A = 10 ** (gain_db / 40)
    al = np.sin(w0) / (2 * q)
    if kind == "peak":
        b = [1 + al * A, -2 * np.cos(w0), 1 - al * A]
        a = [1 + al / A, -2 * np.cos(w0), 1 - al / A]
    elif kind == "hp":
        c = np.cos(w0)
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        a = [1 + al, -2 * c, 1 - al]
    elif kind == "ls":
        c, s2 = np.cos(w0), 2 * np.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * c + s2), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - s2)]
        a = [(A + 1) + (A - 1) * c + s2, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - s2]
    return sg.lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=-1)


def env_of(x, atk_ms, rel_ms):
    e = np.abs(sg.hilbert(x))
    ka, kr = max(1, int(atk_ms / 1000 * SR)), max(1, int(rel_ms / 1000 * SR))
    return np.convolve(e, np.ones(ka) / ka, mode="same"), np.convolve(e, np.ones(kr) / kr, mode="same")


# ---------- ВОКАЛ: HPF, де-эссер, срез мути, чуть меньше ВЧ-хвоста ----------
v = st["vocals"]
v = bq(v, "hp", 95)
v = bq(v, "peak", 300, -2.5, q=1.0)          # муть
sib = sg.sosfilt(sg.butter(4, [5200 / (SR / 2), 9500 / (SR / 2)], btype="band", output="sos"), v)
se = np.abs(sg.hilbert(sib.mean(0)))
se = np.convolve(se, np.ones(int(0.004 * SR)) / int(0.004 * SR), mode="same")
thr = np.percentile(se, 92)
duck = np.clip(thr / (se + 1e-9), 0.35, 1.0)
v = v - sib * (1 - duck)                      # де-эссер: давим только сибилянты
print(f"вокал: HPF 95, -2.5 dB @300, де-эссер до {20*np.log10(duck.min()):.1f} dB")

# ---------- УДАРНЫЕ: панч кика и хлёсткий транзиент ----------
d = st["drums"]
low = sg.sosfilt(sg.butter(4, 110 / (SR / 2), btype="low", output="sos"), d)
fast, slow = env_of(d.mean(0), 3, 45)
tr = np.clip((fast - slow) / (slow + 1e-6), 0, 2.0)
d = d * (1 + 0.55 * tr) + low * 0.35          # атака + вес кика
d = bq(d, "peak", 320, -3.0, q=1.0)
print(f"ударные: транзиент +{20*np.log10(1+0.55*tr.max()):.1f} dB пик, кик +3 dB, -3 dB @320")

# ---------- БАС: HPF, вырез мути, дакинг от кика ----------
b_ = st["bass"]
b_ = bq(b_, "hp", 32)
b_ = bq(b_, "peak", 260, -3.5, q=1.0)
ke = np.abs(sg.hilbert(low.mean(0)))
ke = np.convolve(ke, np.ones(int(0.012 * SR)) / int(0.012 * SR), mode="same")
kn = ke / (np.percentile(ke, 97) + 1e-9)
b_ = b_ * (1 - 0.45 * np.clip(kn, 0, 1))      # sidechain
print("бас: HPF 32, -3.5 dB @260, сайдчейн от кика -45 %")

# ---------- ГАРМОНИЯ: чистка низкой середины и расширение стерео ----------
o = st["other"]
o = bq(o, "hp", 90)
o = bq(o, "peak", 300, -4.0, q=0.9)
mid, side = (o[0] + o[1]) / 2, (o[0] - o[1]) / 2
side_hi = sg.sosfilt(sg.butter(2, 250 / (SR / 2), btype="high", output="sos"), side)
side = side_hi * 1.9 + (side - side_hi) * 0.6   # шире сверху, моно снизу
o = np.vstack([mid + side, mid - side])
print("гармония: HPF 90, -4 dB @300, стерео сверху ×1.9, низ в моно")

# ---------- АВТОМАТИЗАЦИЯ КУЛЬМИНАЦИИ (эталон: пик на 140-175 с) ----------
t = np.arange(L) / SR
climax = np.clip((t - 118) / 26, 0, 1) * np.clip((186 - t) / 8, 0, 1)   # плавно вверх к 144, спад к 186
gl = 10 ** (climax * 2.6 / 20)                                          # до +2.6 dB
d = d * (1 + 0.35 * climax)
o_mid, o_side = (o[0] + o[1]) / 2, (o[0] - o[1]) / 2
o_side = o_side * (1 + 0.45 * climax)                                    # шире к кульминации
o = np.vstack([o_mid + o_side, o_mid - o_side])
sat = lambda x, a: np.tanh(x * (1 + a)) / (1 + a * 0.6)
d = sat(d, 0.5 * climax)
print(f"кульминация: +{2.6:.1f} dB и ×1.45 ширины к 144 с, сатурация ударных")

# ---------- сборка ----------
mix = (v + d + b_ + o) * gl
mix = mix / max(1e-9, np.abs(mix).max()) * 0.9
os.makedirs(f"stems/htdemucs/{tag}", exist_ok=True)
for k, x in [("vocals", v), ("drums", d), ("bass", b_), ("other", o)]:
    sf.write(f"stems/htdemucs/{tag}/{k}.wav", x.T, SR)
sf.write(f"takes/{tag}.wav", mix.T, SR)
s = subprocess.run(["ffmpeg", "-hide_banner", "-i", f"takes/{tag}.wav", "-af", "ebur128", "-f", "null", "-"],
                   capture_output=True, text=True).stderr
I = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", s)[-1])
G = -14.2 - I
subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", f"takes/{tag}.wav",
                "-af", f"volume={G:.2f}dB,alimiter=limit=0.85:attack=5:release=80:level=false",
                "-b:a", "192k", f"takes/{tag}.mp3"])
print(f"готово: takes/{tag}.mp3 ({G:+.1f} dB)")
