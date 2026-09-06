"""Состояние конвейера + оценки времени. Всё пишется в state.json, его читает дашборд."""
import json, os, time, threading

# 🔴 v13.1: корень состояния — ТЕКУЩАЯ ДИРЕКТОРИЯ ПРОГОНА, как в layout.py.
# Было os.path.dirname(__file__) — и state.json писался в lib/ САМОГО СКИЛЛА:
# все прогоны делили один файл, дашборд прогона не видел ни одной стадии,
# а имя чужого прогона (hanbit-ua-v1) висело в шапке чужого борда.
# Замер 2026-08-22: krov-прогон писал состояние в ~/.claude/skills/.../lib/state.json.
ROOT = os.getcwd()
STATE = os.path.join(ROOT, "state.json")
_lock = threading.Lock()

# замеренные/ожидаемые длительности операций, сек
ETA = {"suno": 150, "nano": 35, "seedance": 110, "whisper": 12, "assemble": 25}

# 🔴 ТАРИФЫ ЗАМЕРЕНЫ СПИСАНИЕМ КРЕДИТОВ 2026-08-21, а не взяты из головы.
# Прежние значения (seedance 480p $0.019/с) ВРАЛИ В ШЕСТЬ РАЗ — все отчёты
# о тратах до этой даты занижены. Мерить только так: credits() до → задание →
# credits() после. 200 кредитов = $1.
# замер 2026-08-22 (krov): 9 листов 2K подряд по одному кошельку = 120 кредитов
# → 13.3 кредита = $0.0667 за картинку 2K. Suno-запрос = 12 кредитов = $0.06 (сходится).
COST = {"suno_request": 0.06, "nano": 0.04, "nano_2k": 0.067,
        # 🔴 ЗАМЕР 23.08.2026, окно 13с в режиме reference-video: credits() до
        # 87970.4 → после 87903.2 = 67.2 кредита = 5.17 кред/с = $0.0258/с.
        # Прежние 0.116 ЗАВЫШАЛИ В 4.5 РАЗА: весь прогон krov показывал $61.5,
        # реально ушло около $14. Отчёты о тратах до этой даты не верны.
        "seedance_per_sec_480p": 0.0258,
        "seedance_per_sec_720p": 0.250,
        "refvideo_per_sec_480p": 0.0258,     # тот же замер: режим reference-video
                                             # НЕ дороже обычного i2v, вопреки старой записи
        "kling_motion_control": 0.330}       # замерено: за шот, только 720p+

# 🔴 СТАДИИ — ЭТО v13.3, А НЕ v12. Живой случай 23.08.2026: пользователь смотрел
# на борд чужого прогона и видел «Кейфреймы персонажей», «ГЕЙТ 2 · Клипы 20 секунд»,
# «Перегенерация под аудит» — шаги, которых в v13.3 НЕ СУЩЕСТВУЕТ: единица работы
# окно, покадровой разметки и кейфреймов нет. Список остался от прошлого поколения,
# и борд обещал конвейер, которого нет. Стадии тут обязаны совпадать с порядком
# работы в SKILL.md — это и есть «борд источник правды», иначе правило пустое.
STAGES = [
    ("dna",      "Декод референса",           "refprep: слова, шоты, отрезки + паспорт"),
    ("lyrics",   "Текст песни",               "бюджет слогов → построчно слог в слог"),
    ("song",     "ГЕЙТ · Песня Suno",         "одна генерация + полнота ≥0.92 + WER ≤0.10"),
    ("timeline", "Титры + план окон",         "captions.json → windows.json (~13с)"),
    ("chars",    "Паспорта персонажей",       "cast.json → листы + каст по окнам"),
    ("wave",     "🔴 ГЕЙТ · Волна музыки",    "музыка ≥0.75 · голос ≥0.30 · пик ±10%"),
    ("windows",  "Окна Seedance",             "reference_video + паспорта, ступенями"),
    ("gatewin",  "ГЕЙТ окон",                 "парный судья: наше окно против эталона"),
    ("assemble", "Сборка",                    "титры + песня + финальная карточка"),
    ("final",    "Финальный гейт",            "скоркард против passport-ref"),
]

def _blank():
    return {
        "run": "run", "started_at": time.time(), "updated_at": time.time(),
        "gate": None, "cost_usd": 0.0, "log": [],
        "stages": [{"id": i, "title": t, "note": n, "status": "pending",
                    "started": None, "ended": None, "done": 0, "total": 0, "detail": ""}
                   for i, t, n in STAGES],
        "jobs": [],
    }

def load():
    if not os.path.exists(STATE): return _blank()
    try:
        with open(STATE, encoding="utf-8") as f: return json.load(f)
    except Exception: return _blank()

def save(s):
    s["updated_at"] = time.time()
    tmp = STATE + f".tmp{os.getpid()}"   # 🔴 унікальний tmp: два процеси затирали один одному
    with open(tmp, "w", encoding="utf-8") as f: json.dump(s, f, ensure_ascii=False)
    os.replace(tmp, STATE)

def _stage(s, sid):
    for st in s["stages"]:
        if st["id"] == sid: return st
    # 🔴 новый этап дописывается сам. Раньше падало KeyError и роняло весь прогон
    # ещё до первой генерации — только потому, что этапа не было в списке.
    known = dict((i, (t, n)) for i, t, n in STAGES)
    t, n = known.get(sid, (sid, ""))
    st = {"id": sid, "title": t, "note": n, "status": "pending",
          "started": None, "ended": None, "done": 0, "total": 0, "detail": ""}
    s["stages"].append(st)
    return st

def start(sid, total=0, detail=""):
    with _lock:
        s = load(); st = _stage(s, sid)
        st.update(status="run", started=time.time(), total=total, done=0, detail=detail)
        s["log"].append({"t": time.time(), "lvl": "go", "msg": f"▶ {st['title']}" + (f" — {detail}" if detail else "")})
        save(s)

def done(sid, detail=None):
    with _lock:
        s = load(); st = _stage(s, sid)
        st.update(status="done", ended=time.time())
        if detail: st["detail"] = detail
        el = int(time.time() - (st["started"] or time.time()))
        s["log"].append({"t": time.time(), "lvl": "ok", "msg": f"✔ {st['title']} — {el}s" + (f" · {detail}" if detail else "")})
        save(s)

def fail(sid, why):
    with _lock:
        s = load(); st = _stage(s, sid)
        st.update(status="fail", ended=time.time(), detail=why)
        s["log"].append({"t": time.time(), "lvl": "err", "msg": f"✖ {st['title']} — {why}"})
        save(s)

def tick(sid, done_n=None, detail=None):
    with _lock:
        s = load(); st = _stage(s, sid)
        if done_n is not None: st["done"] = done_n
        else: st["done"] = st.get("done", 0) + 1
        if detail is not None: st["detail"] = detail
        save(s)

def log(msg, lvl="info"):
    with _lock:
        s = load(); s["log"].append({"t": time.time(), "lvl": lvl, "msg": msg}); save(s)

def gate(name, payload=None):
    with _lock:
        s = load(); s["gate"] = {"name": name, "at": time.time(), "payload": payload or {}}
        s["log"].append({"t": time.time(), "lvl": "gate", "msg": f"⏸ ГЕЙТ: {name}"}); save(s)

def spend(usd, what=""):
    with _lock:
        s = load(); s["cost_usd"] = round(s.get("cost_usd", 0.0) + usd, 4)
        if what: s["log"].append({"t": time.time(), "lvl": "cost", "msg": f"$ {usd:.3f} — {what}"})
        save(s)

def job_open(jid, kind, label, eta=None):
    with _lock:
        s = load()
        s["jobs"] = [j for j in s["jobs"] if j["id"] != jid][-60:]
        s["jobs"].append({"id": jid, "kind": kind, "label": label, "state": "queued",
                          "opened": time.time(), "started": time.time(), "ended": None, "eta": eta or ETA.get(kind, 90)})
        save(s)

def job_state(jid, state):
    with _lock:
        s = load()
        for j in s["jobs"]:
            if j["id"] == jid:
                j["state"] = state
                if state in ("success", "fail"): j["ended"] = time.time()
        save(s)

def reset():
    save(_blank())
