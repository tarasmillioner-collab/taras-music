"""ПРОФИЛЬ ИСПОЛНЕНИЯ ПО СТРОКАМ: где песня ускоряется, где давит, где тянет.
usage: python3 delivery.py REF_LINES.json REF_AUDIO  [CLONE_LINES.json CLONE_AUDIO]
Для каждой строки: длительность, слогов/с, громкость относительно среднего, высота,
и классификация: разгон / торможение / нажим / спад.
"""
import sys, os, json, re
import numpy as np, librosa

SR = 22050
HERE = os.path.dirname(os.path.abspath(__file__))
V_UA = set("аеєиіїоуюя")
NUM = {"34": "thirty four", "16": "sixteen", "15": "fifteen"}


def syl(s):
    if re.search(r"[а-яіїєґ]", s.lower()):
        return sum(sum(c in V_UA for c in w.lower()) for w in re.findall(r"[а-яіїєґ'́]+", s.lower()))
    t = re.sub(r"\d+", lambda m: NUM.get(m.group(0), m.group(0)), s.lower())
    return max(1, sum(1 for _ in re.finditer(r"[aeiouy]+", t)))


def profile(lines, audio):
    y, _ = librosa.load(audio, sr=SR, mono=True)
    dur = len(y) / SR
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=512)
    db = 20 * np.log10(rms + 1e-6)
    base = np.median(db[db > db.max() - 30])
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("E3"), fmax=librosa.note_to_hz("C6"),
                            sr=SR, frame_length=2048, hop_length=512)
    rows = []
    for i, (ts, text) in enumerate(lines):
        te = lines[i + 1][0] if i + 1 < len(lines) else min(ts + 4.0, dur)
        if te <= ts:
            continue
        m = (t >= ts) & (t < te)
        if m.sum() < 2:
            continue
        s = syl(text)
        f = f0[m[:len(f0)]] if len(f0) >= m.sum() else f0
        fv = f[~np.isnan(f)] if f is not None and len(f) else np.array([])
        rows.append(dict(i=i + 1, t=round(ts, 1), dur=round(te - ts, 2), syl=s,
                         sps=round(s / (te - ts), 2),
                         lvl=round(float(db[m].mean() - base), 1),
                         pitch=round(float(librosa.hz_to_midi(np.median(fv))), 1) if len(fv) > 5 else None,
                         text=text))
    sps = np.array([r["sps"] for r in rows])
    med = np.median(sps)
    for r in rows:
        tags = []
        if r["sps"] > med * 1.25:
            tags.append("РАЗГОН")
        if r["sps"] < med * 0.75:
            tags.append("ТОРМОЖЕНИЕ")
        if r["lvl"] > 2.0:
            tags.append("НАЖИМ")
        if r["lvl"] < -3.0:
            tags.append("СПАД")
        r["tag"] = " ".join(tags)
    return rows, med


if __name__ == "__main__":
    ref_lines = json.load(open(sys.argv[1]))
    rows, med = profile(ref_lines, sys.argv[2])
    print(f"ЭТАЛОН: медианный темп {med:.2f} слог/с")
    print(f"{'#':>3s} {'t':>6s} {'длит':>5s} {'сл/с':>5s} {'ур':>5s} {'нота':>5s}  строка")
    for r in rows:
        print(f"{r['i']:3d} {r['t']:6.1f} {r['dur']:5.2f} {r['sps']:5.2f} {r['lvl']:+5.1f} "
              f"{str(r['pitch'] or '-'):>5s}  {r['text'][:44]:44s} {r['tag']}")
    json.dump(rows, open(os.path.join(HERE, "delivery_ref.json"), "w"), ensure_ascii=False, indent=1)
    if len(sys.argv) > 4:
        cl, medc = profile(json.load(open(sys.argv[3])), sys.argv[4])
        json.dump(cl, open(os.path.join(HERE, "delivery_clone.json"), "w"), ensure_ascii=False, indent=1)
        print(f"\nКЛОН: медианный темп {medc:.2f} слог/с")
        n = min(len(rows), len(cl))
        dev = [(rows[i]["i"], rows[i]["sps"], cl[i]["sps"], rows[i]["tag"], cl[i]["tag"], rows[i]["text"])
               for i in range(n) if rows[i]["tag"] != cl[i]["tag"]]
        print(f"строк, где исполнение разошлось: {len(dev)} из {n}")
        for i, a, b, ta, tb, tx in dev[:30]:
            print(f"  #{i:3d} эталон {a:4.2f} [{ta or '—':22s}]  клон {b:4.2f} [{tb or '—':22s}]  {tx[:34]}")
