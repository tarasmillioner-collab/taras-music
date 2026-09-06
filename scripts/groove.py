"""ГРУВ вокала: попадает ли певица в карман доли и живая ли ритмика.
usage: python3 groove.py NAME file [NAME file ...]   (файл — микс или вокальный стем)
Меры (сравниваются с эталоном):
  phase_spread — разброс фаз вокальных атак внутри доли (робот кучкуется, живой — оттягивает)
  onbeat_pct   — доля атак ровно на доле
  swing        — смещение атак на слабых долях (оттяжка назад = грув)
  dur_cv       — вариативность длительностей слогов (робот ровный)
  accent_range — разброс силы атак (живой акцентирует)
"""
import sys, os, json
import numpy as np, librosa, scipy.signal as sg

SR = 22050
HERE = os.path.dirname(os.path.abspath(__file__))
BPM = 103.4
BEAT = 60 / BPM


def gfeat(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    yh, _ = librosa.effects.hpss(y)
    # полоса голоса
    b = sg.butter(4, [220 / (SR / 2), 3500 / (SR / 2)], btype="band", output="sos")
    v = sg.sosfilt(b, yh)
    oenv = librosa.onset.onset_strength(y=v, sr=SR, hop_length=256)
    ons = librosa.onset.detect if False else librosa.onset.onset_detect
    t = ons(onset_envelope=oenv, sr=SR, hop_length=256, units="time", delta=0.08, wait=3)
    if len(t) < 20:
        return None
    strength = np.interp(t, librosa.frames_to_time(np.arange(len(oenv)), sr=SR, hop_length=256), oenv)
    ph = (t % BEAT) / BEAT                       # фаза внутри доли
    ph_c = (ph + 0.5) % 1.0 - 0.5                # центрируем: 0 = ровно на доле
    hist, _ = np.histogram(ph, bins=8, range=(0, 1))
    hist = hist / hist.sum()
    d = np.diff(t)
    d = d[(d > 0.05) & (d < 1.5)]
    return {
        "n_onsets": int(len(t)),
        "onsets_per_s": float(len(t) / (len(y) / SR)),
        "phase_spread": float(np.std(ph_c)),                       # разброс относительно доли
        "phase_entropy": float(-(hist * np.log(hist + 1e-9)).sum()),  # равномерность = живость
        "onbeat_pct": float(np.mean(np.abs(ph_c) < 0.12) * 100),
        "swing": float(np.median(ph_c)),                            # + = позади доли (оттяжка)
        "dur_cv": float(np.std(d) / (np.mean(d) + 1e-9)),
        "accent_range": float(np.percentile(strength, 90) / (np.percentile(strength, 30) + 1e-9)),
    }


if __name__ == "__main__":
    R = gfeat(os.path.join(HERE, "full.wav"))
    keys = ["onsets_per_s", "phase_spread", "phase_entropy", "onbeat_pct", "swing", "dur_cv", "accent_range"]
    print(f"{'трек':16s}" + "".join(f"{k[:9]:>11s}" for k in keys) + f"{'ГРУВ':>7s}")
    def gscore(f):
        if f is None:
            return 0.0
        s = (0.25 * np.clip(1 - abs(f["phase_entropy"] - R["phase_entropy"]) / 0.35, 0, 1)
             + 0.20 * np.clip(1 - abs(f["dur_cv"] - R["dur_cv"]) / 0.30, 0, 1)
             + 0.20 * np.clip(1 - abs(f["accent_range"] - R["accent_range"]) / (R["accent_range"] * 0.45), 0, 1)
             + 0.20 * np.clip(1 - abs(f["onbeat_pct"] - R["onbeat_pct"]) / 14.0, 0, 1)
             + 0.15 * np.clip(1 - abs(f["onsets_per_s"] - R["onsets_per_s"]) / (R["onsets_per_s"] * 0.35), 0, 1))
        return float(s * 10)
    print(f"{'ЭТАЛОН':16s}" + "".join(f"{R[k]:11.3f}" for k in keys) + f"{gscore(R):7.2f}")
    out = {}
    for n, p in zip(sys.argv[1::2], sys.argv[2::2]):
        f = gfeat(p)
        if f is None:
            print(f"{n:16s}  мало атак")
            continue
        out[n] = {"raw": f, "groove": round(gscore(f), 2)}
        print(f"{n:16s}" + "".join(f"{f[k]:11.3f}" for k in keys) + f"{gscore(f):7.2f}")
    json.dump(out, open(os.path.join(HERE, "groove.json"), "w"), ensure_ascii=False, indent=1)
