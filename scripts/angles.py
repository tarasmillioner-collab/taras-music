"""УГЛЫ, КОТОРЫЕ НЕ СМОТРЕЛИ.
1) лад: мажор или минор (Krumhansl-Schmuckler) — «грустно» живёт здесь
2) первые 15 секунд отдельно: что играет, как ярко, как быстро
3) тесситура вокала: в какой октаве поёт и насколько это «светло»
4) время до первого слова и до первого удара
5) интонация первой фразы: вверх или вниз
6) мажорная терция против минорной: прямой замер по хроме
7) яркость (центроид) в начале против середины
8) плотность слов в первые 20 с
usage: python3 angles.py NAME file [NAME file ...]
"""
import sys, os, json
import numpy as np, librosa, scipy.signal as sg

SR = 22050
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_of(y, sr=SR):
    yh, _ = librosa.effects.hpss(y)
    ch = librosa.feature.chroma_cqt(y=yh, sr=sr).mean(axis=1)
    best = []
    for i in range(12):
        best.append((np.corrcoef(ch, np.roll(MAJ, i))[0, 1], NAMES[i] + " major", i, "maj"))
        best.append((np.corrcoef(ch, np.roll(MIN, i))[0, 1], NAMES[i] + " minor", i, "min"))
    best.sort(reverse=True)
    corr, name, root, mode = best[0]
    # прямое сравнение: сколько энергии на большой терции против малой
    third_maj = ch[(root + 4) % 12]
    third_min = ch[(root + 3) % 12]
    return name, float(corr), float(third_maj / (third_min + 1e-9)), ch


def angles(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    dur = len(y) / SR
    a = {}
    kname, kcorr, third_ratio, ch = key_of(y)
    a["key"] = kname
    a["key_corr"] = round(kcorr, 3)
    a["maj3_over_min3"] = round(third_ratio, 2)

    intro = y[:int(15 * SR)]
    kname_i, _, tr_i, _ = key_of(intro)
    a["key_intro"] = kname_i
    a["maj3_intro"] = round(tr_i, 2)

    # яркость: начало против середины
    cen = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=512)[0]
    t = librosa.frames_to_time(np.arange(len(cen)), sr=SR, hop_length=512)
    a["bright_0_15"] = round(float(np.median(cen[t < 15])))
    a["bright_mid"] = round(float(np.median(cen[(t > 80) & (t < 120)])))
    a["bright_ratio"] = round(a["bright_0_15"] / max(a["bright_mid"], 1), 2)

    # энергия начала
    def db(x):
        return 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)
    a["lvl_0_10"] = round(db(y[:int(10 * SR)]) - db(y), 1)

    # время до первого звука и до первого удара
    r = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    tr_ = librosa.frames_to_time(np.arange(len(r)), sr=SR, hop_length=256)
    thr = np.percentile(r, 60)
    first = tr_[np.argmax(r > thr * 0.35)]
    a["t_first_sound"] = round(float(first), 2)
    _, yp = librosa.effects.hpss(y)
    on = librosa.onset.onset_detect(y=yp, sr=SR, units="time", delta=0.12)
    perc = [x for x in on if x > 0.3]
    a["t_first_beat"] = round(float(perc[0]), 2) if perc else None

    # тесситура вокала в первые 20 с и в целом
    b = sg.butter(4, [180 / (SR / 2), 2200 / (SR / 2)], btype="band", output="sos")
    v = sg.sosfilt(b, y)
    f0, _, _ = librosa.pyin(v, fmin=librosa.note_to_hz("E3"), fmax=librosa.note_to_hz("C6"),
                            sr=SR, frame_length=2048, hop_length=256)
    tf = librosa.frames_to_time(np.arange(len(f0)), sr=SR, hop_length=256)
    ok = ~np.isnan(f0)
    if ok.sum() > 50:
        m = librosa.hz_to_midi(f0[ok])
        a["voice_median_note"] = librosa.midi_to_note(int(round(np.median(m))))
        a["voice_median_midi"] = round(float(np.median(m)), 1)
        e = ok & (tf < 20)
        if e.sum() > 30:
            me = librosa.hz_to_midi(f0[e])
            a["voice_intro_note"] = librosa.midi_to_note(int(round(np.median(me))))
            a["voice_intro_midi"] = round(float(np.median(me)), 1)
            # контур первой фразы: наклон высоты за первые 20 с
            xs = tf[e]
            a["intro_pitch_slope"] = round(float(np.polyfit(xs, me, 1)[0]), 2)
    # плотность событий в первые 20 с
    onall = librosa.onset.onset_detect(y=y, sr=SR, units="time", delta=0.08)
    a["events_0_20"] = round(float(sum(1 for x in onall if x < 20) / 20), 2)
    a["events_mid"] = round(float(sum(1 for x in onall if 80 < x < 120) / 40), 2)
    return a


if __name__ == "__main__":
    keys = ["key", "key_intro", "maj3_over_min3", "maj3_intro", "bright_0_15", "bright_ratio",
            "lvl_0_10", "t_first_sound", "t_first_beat", "voice_intro_note", "voice_median_note",
            "intro_pitch_slope", "events_0_20", "events_mid"]
    R = angles(os.path.join(HERE, "full.wav"))
    out = {"REF": R}
    for n, p in zip(sys.argv[1::2], sys.argv[2::2]):
        out[n] = angles(p)
    print(f"{'угол':22s}" + "".join(f"{n[:11]:>13s}" for n in out))
    for k in keys:
        row = "".join(f"{str(out[n].get(k, '-')):>13s}" for n in out)
        print(f"{k:22s}{row}")
    json.dump(out, open(os.path.join(HERE, "angles.json"), "w"), ensure_ascii=False, indent=1, default=float)
