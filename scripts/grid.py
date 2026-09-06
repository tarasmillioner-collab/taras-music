"""СЕТКА: ложатся ли слова на доли такта (песня) или куда придётся (начитка).
usage: python3 grid.py NAME audio.mp3 lines.json [NAME audio lines ...]
Меры:
  on_beat   — доля вокальных атак в ±60 мс от доли (песня попадает, начитка нет)
  on_strong — доля атак на сильных долях такта (1 и 3)
  bar_fit   — доля строк, чья длина укладывается в целое число полутактов
  phrase_bars — средняя длина строки в тактах и разброс
"""
import sys, os, json
import numpy as np, librosa, scipy.signal as sg

SR = 22050
BPM = 103.4
BEAT = 60 / BPM
BAR = 4 * BEAT
HERE = os.path.dirname(os.path.abspath(__file__))


def grid(audio, lines=None):
    y, _ = librosa.load(audio, sr=SR, mono=True)
    dur = len(y) / SR
    yh, _ = librosa.effects.hpss(y)
    b = sg.butter(4, [200 / (SR / 2), 3500 / (SR / 2)], btype="band", output="sos")
    v = sg.sosfilt(b, yh)
    oenv = librosa.onset.onset_strength(y=v, sr=SR, hop_length=256)
    t = librosa.onset.onset_detect(onset_envelope=oenv, sr=SR, hop_length=256,
                                   units="time", delta=0.07, wait=2)
    st = np.interp(t, librosa.frames_to_time(np.arange(len(oenv)), sr=SR, hop_length=256), oenv)
    # сетка от первого сильного события
    off = t[0] % BEAT if len(t) else 0.0
    ph = ((t - off) % BEAT) / BEAT
    ph_c = (ph + 0.5) % 1.0 - 0.5
    on_beat = float(np.mean(np.abs(ph_c) < 0.105) * 100)          # ±60 мс
    bar_ph = ((t - off) % BAR) / BAR
    strong = (np.abs(bar_ph - 0.0) < 0.06) | (np.abs(bar_ph - 0.5) < 0.06)
    on_strong = float(np.mean(strong) * 100)
    # сильные атаки (верхняя треть по энергии) отдельно — они важнее
    thr = np.percentile(st, 67)
    hard = np.abs(ph_c[st >= thr]) < 0.105
    on_beat_hard = float(np.mean(hard) * 100) if hard.size else 0.0
    res = dict(onsets=len(t), on_beat=round(on_beat, 1), on_strong=round(on_strong, 1),
               on_beat_hard=round(on_beat_hard, 1))
    if lines:
        L = [x for x in lines if x[0] is not None]
        d = [L[i + 1][0] - L[i][0] for i in range(len(L) - 1)]
        d = np.array([x for x in d if 0.3 < x < 12])
        bars = d / BAR
        near = np.abs(bars - np.round(bars * 2) / 2)                # кратность полутакту
        res["phrase_bars_med"] = round(float(np.median(bars)), 2)
        res["bar_fit"] = round(float(np.mean(near < 0.12) * 100), 1)
        res["phrase_bars_sd"] = round(float(np.std(bars)), 2)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    print(f"{'трек':10s}{'атак':>6s}{'на доле':>9s}{'сильн.':>8s}{'акценты':>9s}{'фраза,такт':>12s}{'в сетке':>9s}{'разброс':>9s}")
    out = {}
    i = 0
    while i < len(a):
        name, audio = a[i], a[i + 1]
        lines = None
        if i + 2 < len(a) and a[i + 2].endswith(".json"):
            try:
                lines = [[x["t"], x["line"]] for x in json.load(open(a[i + 2]))]
            except Exception:
                lines = [[x[0], x[1]] for x in json.load(open(a[i + 2]))]
            i += 3
        else:
            i += 2
        r = grid(audio, lines)
        out[name] = r
        print(f"{name:10s}{r['onsets']:6d}{r['on_beat']:8.1f}%{r['on_strong']:7.1f}%{r['on_beat_hard']:8.1f}%"
              f"{str(r.get('phrase_bars_med','—')):>12s}{str(r.get('bar_fit','—')):>8s}%{str(r.get('phrase_bars_sd','—')):>9s}")
    json.dump(out, open(os.path.join(HERE, "grid.json"), "w"), ensure_ascii=False, indent=1)
