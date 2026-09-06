"""Построчная проверка: каждая строка текста ищется в транскрипте (whisper medium) по лучшему окну.
usage: LYRICS=lyrics_uk_v3.txt python3 lines_check.py takes/take_G_1.mp3 ...
Строка считается спетой при сходстве ≥0.75, проглоченной при <0.55."""
import sys, os, re, json, difflib
import whisper
os.chdir(os.path.dirname(os.path.abspath(__file__)))
NUM = {"34": "тридцять чотири", "16": "шістнадцять", "15": "п'ятнадцять", "50": "п'ятдесят", "10": "десять", "5": "п'ять", "3": "три", "6": "шостий", "2": "дві"}
def norm(s):
    s = s.lower().replace("’", "'").replace("ʼ", "'"); s = re.sub(r"\d+", lambda m: NUM.get(m.group(0), m.group(0)), s)
    return re.findall(r"[а-яіїєґa-z']+", s)
lyr = os.environ.get("LYRICS", "lyrics_uk_v3.txt")
lines = [l for l in open(lyr, encoding="utf-8").read().splitlines() if l.strip() and not l.startswith("[")]
model = whisper.load_model("medium")
summary = {}
for f in sys.argv[1:]:
    cache = f.replace(".mp3", "_medium.json")
    if os.path.exists(cache): r = json.load(open(cache))
    else:
        r = model.transcribe(f, language="uk", word_timestamps=True, verbose=False); json.dump(r, open(cache, "w"), ensure_ascii=False)
    words = [(w["word"].strip(), w["start"]) for s in r["segments"] for w in s.get("words", [])]
    hyp = [(x, t) for w, t in words for x in norm(w)]
    H = [x for x, _ in hyp]
    res = []; pos = 0
    for ln in lines:
        L = norm(ln); n = len(L); best = (0, pos, None)
        lo = max(0, pos - 6); hi = min(len(H) - 1, pos + 60)
        for i in range(lo, hi + 1):
            for span in (n, n + 1, max(1, n - 1)):
                seg = H[i:i + span]
                if not seg: continue
                sc = difflib.SequenceMatcher(None, " ".join(L), " ".join(seg)).ratio()
                if sc > best[0]: best = (sc, i, hyp[i][1])
        sc, i, t = best
        if sc >= 0.55: pos = i + max(1, n - 1)
        res.append({"line": ln, "score": round(sc, 2), "t": round(t, 1) if t is not None else None, "heard": " ".join(H[i:i + n])})
    ok = sum(r_["score"] >= 0.75 for r_ in res); bad = [r_ for r_ in res if r_["score"] < 0.55]; so = [r_ for r_ in res if 0.55 <= r_["score"] < 0.75]
    summary[f] = {"sung": ok, "total": len(lines), "swallowed": len(bad), "garbled": len(so)}
    print(f"== {f}: спето чисто {ok}/{len(lines)}, смазано {len(so)}, проглочено {len(bad)}")
    for r_ in bad: print(f"   ✗ {r_['score']:.2f} {r_['line']}  ← «{r_['heard']}»")
    for r_ in so: print(f"   ~ {r_['score']:.2f} {r_['line']}  ← «{r_['heard']}»")
    json.dump(res, open(f.replace(".mp3", "_lines.json"), "w"), ensure_ascii=False, indent=0)
json.dump(summary, open("lines_summary.json", "w"), ensure_ascii=False, indent=1)
