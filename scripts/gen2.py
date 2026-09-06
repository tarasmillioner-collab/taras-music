"""Генератор v2: полный набор ручек Suno на kie.ai.
  persona  — снять персону с трека (голос и характер) и переиспользовать
  cover    — upload-cover с audioWeight / styleWeight / weirdnessConstraint / personaId

usage:
  python3 gen2.py cover TAG                       # веса берутся из env
  python3 gen2.py persona TASKID AUDIOID NAME     # создать персону
env: SEED, LYRICS, DUR, STYLE, AW, SW, WC, PERSONA, PMODEL, NEG, MODEL
"""
import sys, os, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                      # своя копия kie.py рядом со скриптом
sys.path.append('/Users/taras/.claude/skills/vsl-song-clone_v13/lib')   # запасной путь
import kie
os.chdir(os.path.dirname(os.path.abspath(__file__)))
mode = sys.argv[1]
CACHE = '.upload_cache.json'


def up(path):
    c = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if path not in c:
        c[path] = kie.upload(path, "audio")
        json.dump(c, open(CACHE, 'w'))
    return c[path]


def wait(tid, tag=None, save=True):
    t0 = time.time()
    while time.time() - t0 < 1800:
        time.sleep(10)
        d = (kie._http("GET", f"{kie.SUNO_INFO}?taskId={tid}").get("data")) or {}
        st = d.get("status")
        print(st, int(time.time() - t0), flush=True)
        if st == "SUCCESS":
            takes = (d.get("response") or {}).get("sunoData") or []
            out = []
            for i, t in enumerate(takes, 1):
                out.append({"audioId": t.get("id"), "dur": t.get("duration")})
                if save and tag:
                    u = t.get("audioUrl") or t.get("sourceAudioUrl")
                    dest = f"takes/take_{tag}_{i}.mp3"
                    kie.download(u, dest)
                    sz = os.path.getsize(dest)
                    print(f"saved {dest} {t.get('duration')} {t.get('id')} {sz} bytes", flush=True)
                    if sz < 3_000_000:
                        print("!! ФАЙЛ ОБРЕЗАН, перекачиваю", flush=True)
                        kie.download(u, dest)
            return out
        if st in ("CREATE_TASK_FAILED", "GENERATE_AUDIO_FAILED", "SENSITIVE_WORD_ERROR"):
            print("FAIL", st, d.get("errorMessage"), flush=True)
            return None
    print("timeout", flush=True)
    return None


if mode == "cover":
    tag = sys.argv[2]
    os.makedirs("takes", exist_ok=True)
    lyrics = open(os.environ.get("LYRICS", "lyrics_uk_v17.txt"), encoding="utf-8").read().strip()
    body = {
        "uploadUrl": up(os.environ.get("SEED", "../full.mp3")),
        "prompt": lyrics[:5000],
        "customMode": True,
        "instrumental": False,
        "model": os.environ.get("MODEL", "V5_5"),
        "style": os.environ.get("STYLE", "")[:1000],
        "title": os.environ.get("TITLE", "UA clone"),
        "vocalGender": "f",
        "duration": int(os.environ.get("DUR", "210")),
        "callBackUrl": kie.CALLBACK,
    }
    for env, key in [("AW", "audioWeight"), ("SW", "styleWeight"), ("WC", "weirdnessConstraint")]:
        if os.environ.get(env):
            body[key] = float(os.environ[env])
    if os.environ.get("PERSONA"):
        body["personaId"] = os.environ["PERSONA"]
        if os.environ.get("PMODEL"):
            body["personaModel"] = os.environ["PMODEL"]
    if os.environ.get("NEG"):
        body["negativeTags"] = os.environ["NEG"]
    print("ручки:", {k: v for k, v in body.items() if k in
                     ("audioWeight", "styleWeight", "weirdnessConstraint", "personaId", "personaModel",
                      "negativeTags", "duration", "model")}, flush=True)
    r = kie._http("POST", kie.BASE + "/api/v1/generate/upload-cover", body)
    tid = (r.get("data") or {}).get("taskId")
    print("task", tid, r if not tid else "", flush=True)
    if tid:
        res = wait(tid, tag)
        print("TASKID", tid, flush=True)
        if res:
            print("AUDIOIDS", json.dumps(res), flush=True)

elif mode == "persona":
    task_id, audio_id, name = sys.argv[2], sys.argv[3], sys.argv[4]
    body = {
        "taskId": task_id,
        "audioId": audio_id,
        "name": name,
        "description": os.environ.get("PDESC",
            "Warm breathy female voice, intimate conversational talk-singing, storytelling delivery, "
            "acoustic folk-pop, fingerpicked guitar, soft muted drums, bright major harmony"),
        "vocalStart": float(os.environ.get("PSTART", "24")),
        "vocalEnd": float(os.environ.get("PEND", "50")),
    }
    if os.environ.get("PSTYLE"):
        body["style"] = os.environ["PSTYLE"]
    r = kie._http("POST", kie.BASE + "/api/v1/generate/generate-persona", body)
    print(json.dumps(r, ensure_ascii=False), flush=True)
