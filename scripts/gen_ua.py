"""usage: python3 gen_ua.py cover TAG AUDIO_WEIGHT | python3 gen_ua.py gen TAG
Пишет takes/take_TAG_1.mp3, takes/take_TAG_2.mp3"""
import sys, json, time, os
sys.path.insert(0, '/Users/taras/.claude/skills/vsl-song-clone_v13/lib')
import kie
os.chdir(os.path.dirname(os.path.abspath(__file__)))
mode, tag = sys.argv[1], sys.argv[2]
os.makedirs("takes", exist_ok=True)
lyrics = open(os.environ.get('LYRICS','lyrics_uk.txt'), encoding='utf-8').read().strip()
DUR = int(os.environ.get("DUR", "194"))
STYLE_COVER = os.environ.get('STYLE') or ("Ukrainian female vocal, warm breathy tired storytelling voice, intimate conversational talk-singing, "
               "clear diction, sung in Ukrainian, acoustic narrative folk-pop ballad, 103 bpm, no rap, no autotune, no chorus")
STYLE_GEN = ("acoustic narrative folk-pop, confessional singer-songwriter ballad, 103 bpm, F major, "
             "warm breathy female vocal singing in Ukrainian, intimate conversational talk-singing, clear diction, "
             "close dry mono lead vocal upfront, fingerpicked acoustic guitar eighth-note arpeggio, soft piano block chords, "
             "muted kick with finger snaps, sixteenth-note shaker, sub sine bass on root notes, "
             "sparse intro voice and guitar only, drums enter on second verse, full arrangement at the emotional peak, "
             "everything drops to voice and guitar for the final line, clean modern commercial pop mix, "
             "no chorus, through-composed storytelling, no instrumental breaks, no rap, no autotune")
CACHE = '.upload_cache.json'
if mode == "cover":
    w = float(sys.argv[3])
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    key = os.environ.get('SEED', '../full.mp3')
    if key not in cache:
        cache[key] = kie.upload(key, "audio"); json.dump(cache, open(CACHE, 'w'))
    url = cache[key]; print("seed", url, flush=True)
    instr = os.environ.get('INSTR') == '1'
    body = {"uploadUrl": url, "prompt": "" if instr else lyrics[:5000], "customMode": True, "instrumental": instr, "model": os.environ.get("MODEL", "V5_5"),
            "style": STYLE_COVER, "title": "Сільпо (UA clone)", "vocalGender": "f", "audioWeight": w, "duration": DUR,
            "callBackUrl": kie.CALLBACK}
    r = kie._http("POST", kie.BASE + "/api/v1/generate/upload-cover", body)
else:
    body = {"customMode": True, "instrumental": False, "model": "V5_5", "prompt": lyrics[:5000],
            "style": STYLE_GEN[:1000], "title": "Сільпо (UA gen)", "vocalGender": "f", "duration": DUR,
            "callBackUrl": kie.CALLBACK}
    r = kie._http("POST", kie.SUNO_CREATE, body)
tid = ((r.get("data") or {}).get("taskId")); print("task", tid, r if not tid else "", flush=True)
if not tid: sys.exit(1)
t0 = time.time()
while time.time() - t0 < 1800:
    time.sleep(10)
    d = (kie._http("GET", f"{kie.SUNO_INFO}?taskId={tid}").get("data")) or {}
    st = d.get("status"); print(st, int(time.time() - t0), flush=True)
    if st == "SUCCESS":
        takes = (d.get("response") or {}).get("sunoData") or []
        for i, t in enumerate(takes, 1):
            u = t.get("audioUrl") or t.get("sourceAudioUrl"); dest = f"takes/take_{tag}_{i}.mp3"
            kie.download(u, dest); print("saved", dest, t.get("duration"), t.get("id"), flush=True)
        break
    if st in ("CREATE_TASK_FAILED", "GENERATE_AUDIO_FAILED", "SENSITIVE_WORD_ERROR"):
        print("FAIL", st, d.get("errorMessage"), flush=True); break
