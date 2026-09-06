"""Скоринг дублей против ДНК эталона. usage: python3 score_ua.py takes/take_*.mp3"""
import sys, re, json, glob, subprocess, os
import numpy as np, librosa
from faster_whisper import WhisperModel
os.chdir(os.path.dirname(os.path.abspath(__file__)))
REF = dict(dur=194.4, bpm=103.4, syl_min=198, voiced_pct=70.7, wpm=146, gaps_gt1=5, lufs=-14.2)
V = set("аеєиіїоуюя")
def syl(w): return sum(ch in V for ch in w.lower())
NUM = {"34":"тридцять чотири","16":"шістнадцять","15":"п'ятнадцять","50":"п'ятдесят","10":"десять","5":"п'ять","3":"три","6":"шостий"}
def norm(s):
    s = s.lower().replace("’", "'").replace("ʼ", "'").replace("-", " ")
    s = re.sub(r"\d+", lambda m: NUM.get(m.group(0), m.group(0)), s)
    s = s.replace("півроку", "пів року").replace("чолощок", "чоло щок")
    return re.findall(r"[а-яіїєґa-z']+", s)
def wer(r, h):
    d = np.zeros((len(r)+1, len(h)+1), dtype=int); d[:, 0] = range(len(r)+1); d[0, :] = range(len(h)+1)
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1, d[i-1, j-1]+(r[i-1] != h[j-1]))
    return d[len(r), len(h)]/max(1, len(r))
lines = [l for l in open(os.environ.get("LYRICS","lyrics_uk.txt"), encoding="utf-8").read().splitlines() if l.strip() and not l.startswith("[")]
ref_words = norm(" ".join(lines))
ANCH = {"сільпо": 0.0, "кореянк": 82.7, "тижн": 122.5, "червень": 145.9, "посиланн": 182.5}
SCALE = float(os.environ.get("REFSCALE","1.0"))
ANCH = {k: v*SCALE for k, v in ANCH.items()}
m = WhisperModel("small", device="cpu", compute_type="int8")
out = []
for f in sorted(sys.argv[1:]) or sorted(glob.glob("takes/take_*.mp3")):
    y, sr = librosa.load(f, sr=22050, mono=True); dur = len(y)/sr
    _, yp = librosa.effects.hpss(y)
    on = librosa.onset.onset_strength(y=yp, sr=sr)
    bpm = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=on, sr=sr))[0])
    lufs = subprocess.run(["ffmpeg", "-hide_banner", "-i", f, "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True).stderr
    mI = (re.findall(r"I:\s+(-?[\d.]+) LUFS", lufs) or [None])[-1]; lra = (re.findall(r"LRA:\s+([\d.]+) LU", lufs) or [None])[-1]
    segs, _ = m.transcribe(f, language="uk", word_timestamps=True); segs = list(segs)
    words = [w for s in segs for w in (s.words or [])]
    hyp = norm(" ".join(s.text for s in segs))
    w = wer(ref_words, hyp)
    last = segs[-1].end if segs else 0
    gaps = [segs[i+1].start-segs[i].end for i in range(len(segs)-1)]
    voiced = sum(min(s.end, dur)-s.start for s in segs)/dur*100
    nsyl = sum(syl(x.word) for x in words); sylmin = nsyl/(last/60) if last else 0
    wpm = len(words)/(last/60) if last else 0
    # потерянные строки: строка считается спетой, если >=60% её слов есть в окне гипотезы
    hs = set(hyp); lost = []
    for ln in lines:
        lw = norm(ln)
        if lw and sum(x in hs for x in lw)/len(lw) < 0.6: lost.append(ln)
    def at(pat):
        for s in segs:
            if re.search(pat, s.text.lower()): return round(s.start, 1)
        return None
    anchors = {k: at(k) for k in ANCH}
    drift = {k: (None if anchors[k] is None else round(anchors[k]-ANCH[k], 1)) for k in ANCH}
    rec = dict(file=f, dur=round(dur, 1), last_vocal=round(last, 1), bpm=round(bpm, 1), lufs=float(mI) if mI else None,
               lra=float(lra) if lra else None, wer=round(float(w), 3), lost_lines=len(lost), lost=lost[:8],
               syl_min=round(sylmin), wpm=round(wpm), voiced_pct=round(voiced, 1), gaps_gt1=int(sum(g > 1 for g in gaps)),
               max_gap=round(max(gaps), 1) if gaps else 0, anchors=anchors, drift=drift)
    out.append(rec)
    json.dump([dict(s=round(s.start, 2), e=round(s.end, 2), t=s.text.strip()) for s in segs],
              open(f.replace(".mp3", "_whisper.json"), "w"), ensure_ascii=False, indent=0)
    print(json.dumps({k: v for k, v in rec.items() if k != "lost"}, ensure_ascii=False), flush=True)
    if lost: print("   потеряно:", " | ".join(lost[:6]), flush=True)
json.dump(out, open("scores.json", "w"), ensure_ascii=False, indent=1)
