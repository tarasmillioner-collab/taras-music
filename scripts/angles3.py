"""УГЛЫ 3: пение против начитки на уровне слога и такта.
1) затакт: начинается ли строка ДО сильной доли или ровно на ней
2) гласные против согласных: в пении гласные тянутся, в речи нет
3) инструментальные заполнения между строками
4) ритмический рисунок: ровные восьмые или смешанные длительности
5) атака слога: мягкая (пение) или жёсткая (речь)
usage: python3 angles3.py NAME audio lines.json [...]
"""
import sys, os, json
import numpy as np, librosa, scipy.signal as sg

SR = 22050
BPM = 103.4
BEAT = 60 / BPM
BAR = 4 * BEAT
HERE = os.path.dirname(os.path.abspath(__file__))


def look(audio, lines):
    y, _ = librosa.load(audio, sr=SR, mono=True)
    yh, yp = librosa.effects.hpss(y)
    dur = len(y) / SR
    b = sg.butter(4, [200 / (SR / 2), 3500 / (SR / 2)], btype="band", output="sos")
    v = sg.sosfilt(b, yh)
    r = {}

    # сетка тактов от первой доли
    on_p = librosa.onset.onset_strength(y=yp, sr=SR)
    _, beats = librosa.beat.beat_track(onset_envelope=on_p, sr=SR, trim=False)
    bt = librosa.frames_to_time(beats, sr=SR)
    off = bt[0] if len(bt) else 0.0

    # --- 1. ЗАТАКТ: фаза начала строки внутри такта ---
    L = [x[0] for x in lines if x[0] is not None]
    ph = ((np.array(L) - off) % BAR) / BAR
    # затакт = строка стартует в последней четверти такта (перед сильной долей)
    r["line_pickup_pct"] = round(float(np.mean(ph > 0.75) * 100), 1)
    r["line_on_downbeat_pct"] = round(float(np.mean((ph < 0.10) | (ph > 0.97)) * 100), 1)
    r["line_phase_sd"] = round(float(np.std(((ph + 0.5) % 1) - 0.5)), 3)

    # --- 2. ГЛАСНЫЕ ПРОТИВ СОГЛАСНЫХ ---
    zcr = librosa.feature.zero_crossing_rate(v, frame_length=1024, hop_length=256)[0]
    rms = librosa.feature.rms(y=v, frame_length=1024, hop_length=256)[0]
    flat = librosa.feature.spectral_flatness(y=v, n_fft=1024, hop_length=256)[0]
    live = rms > np.percentile(rms, 45)
    vow = live & (zcr < np.percentile(zcr[live], 55)) & (flat < np.percentile(flat[live], 60))
    con = live & ~vow
    r["vowel_time_pct"] = round(float(vow.sum() / max(live.sum(), 1) * 100), 1)
    # средняя длина непрерывного гласного участка
    runs, s = [], None
    for i, x in enumerate(vow):
        if x and s is None:
            s = i
        elif not x and s is not None:
            runs.append((i - s) * 256 / SR)
            s = None
    r["vowel_run_ms"] = round(float(np.median(runs) * 1000), 0) if runs else None
    r["vowel_long_pct"] = round(float(np.mean(np.array(runs) > 0.20) * 100), 1) if runs else None

    # --- 3. ИНСТРУМЕНТАЛЬНЫЕ ЗАПОЛНЕНИЯ между строками ---
    voiced = rms > np.percentile(rms, 55)
    t = np.arange(len(rms)) * 256 / SR
    gaps = []
    s = None
    for i, x in enumerate(voiced):
        if not x and s is None:
            s = i
        elif x and s is not None:
            if (i - s) * 256 / SR > 0.45:
                gaps.append((t[s], t[i]))
            s = None
    inst = 0
    if gaps:
        oi = librosa.onset.onset_strength(y=yp, sr=SR, hop_length=256)
        ti = np.arange(len(oi)) * 256 / SR
        for a, bnd in gaps:
            m = (ti >= a) & (ti < bnd)
            if m.any() and oi[m].max() > np.percentile(oi, 80):
                inst += 1
    r["gaps"] = len(gaps)
    r["fills_in_gaps_pct"] = round(inst / max(len(gaps), 1) * 100, 1)

    # --- 4. РИТМИЧЕСКИЙ РИСУНОК: разнообразие длительностей между атаками ---
    on = librosa.onset.onset_detect(y=v, sr=SR, units="time", delta=0.07, wait=2)
    d = np.diff(on)
    d = d[(d > 0.08) & (d < 1.2)]
    if len(d) > 20:
        sub = d / (BEAT / 4)                       # в шестнадцатых
        r["subdiv_med"] = round(float(np.median(sub)), 2)
        r["subdiv_sd"] = round(float(np.std(sub)), 2)
        r["subdiv_variety"] = round(float(len(np.unique(np.round(sub))) ), 0)
    # --- 5. АТАКА СЛОГА ---
    env = np.abs(sg.hilbert(v))
    atk = []
    for x in on:
        i0 = int(max(0, (x - 0.02) * SR))
        i1 = int(min(len(env) - 1, (x + 0.09) * SR))
        seg = env[i0:i1]
        if len(seg) > 30 and seg.max() > 1e-6:
            p = int(np.argmax(seg))
            atk.append(p / SR * 1000)
    r["attack_ms"] = round(float(np.median(atk)), 1) if atk else None
    return r


if __name__ == "__main__":
    a = sys.argv[1:]
    out = {}
    for i in range(0, len(a), 3):
        lines = json.load(open(a[i + 2]))
        lines = [[x["t"], x.get("line", "")] if isinstance(x, dict) else x for x in lines]
        out[a[i]] = look(a[i + 1], lines)
    keys = list(next(iter(out.values())).keys())
    print(f"{'угол':24s}" + "".join(f"{n:>12s}" for n in out))
    for k in keys:
        print(f"{k:24s}" + "".join(f"{str(out[n].get(k, '—')):>12s}" for n in out))
    json.dump(out, open(os.path.join(HERE, "angles3.json"), "w"), ensure_ascii=False, indent=1)
