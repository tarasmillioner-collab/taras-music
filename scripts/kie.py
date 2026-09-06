"""Клиент kie.ai: Suno (песня), Seedance 2 mini (видео), nano-banana-2 (кадры).
Контракт проверен по рабочему коду и живым логам."""
import json, os, re, time, threading, urllib.request, urllib.error, subprocess
import pipe

BASE        = "https://api.kie.ai"
JOBS_CREATE = BASE + "/api/v1/jobs/createTask"
JOBS_INFO   = BASE + "/api/v1/jobs/recordInfo"
SUNO_CREATE = BASE + "/api/v1/generate"
SUNO_INFO   = BASE + "/api/v1/generate/record-info"
CREDIT      = BASE + "/api/v1/chat/credit"
UPLOAD      = "https://kieai.redpandaai.co/api/file-stream-upload"
CALLBACK    = "https://example.com/kie-callback"      # обязателен, иначе 422

MAX_PARALLEL = 8          # потолок аккаунта
SUBMIT_GAP   = 1.2        # лимит 20 запросов / 10s
POLL_EVERY   = 5         # 🔴 было 10: на 20 джобах прогона ~1.5 мин чистого простоя
POLL_TIMEOUT = 1500
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"

_slots = threading.Semaphore(MAX_PARALLEL)
_submit = threading.Lock()


def key():
    for line in open(os.path.expanduser("~/.claude/secrets.env"), encoding="utf-8"):
        if line.startswith("KIE_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("KIE_API_KEY не найден в ~/.claude/secrets.env")


def _http(method, url, body=None, tries=3):
    data = json.dumps(body).encode() if body is not None else None
    for a in range(tries):
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": "Bearer " + key(), "Content-Type": "application/json", "User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            txt = e.read().decode()[:300]
            # 🔴 422 тоже ретраим: SKILL обещал «422 временный», а код v12 кидал
            # RuntimeError с первого раза — живые сабмиты падали зря.
            if e.code in (422, 429, 500, 502, 503) and a < tries - 1:
                time.sleep(3 * (a + 1)); continue
            raise RuntimeError(f"HTTP {e.code} {url}: {txt}")
        except Exception:
            if a < tries - 1: time.sleep(3); continue
            raise
    raise RuntimeError("unreachable")


def credits():
    return float(_http("GET", CREDIT).get("data") or 0)


def download(url, dest, tries=3):
    """🔴 ФАЙЛ НА ДИСКЕ ≠ ФАЙЛ, КОТОРЫЙ ОТКРОЕТСЯ. Прогон garlic 24.08.2026:
    окно 4 скачалось ровно 2 048 000 байт — обрыв на границе буфера. Размер
    больше 1 КБ, curl вернул 0, `diskgate` отчитался «критичних 0», и падение
    приехало на ТРИ шага позже, в assemble_part, сырым
    `ValueError: could not convert string to float: ''` из ffprobe — из него
    вообще не видно, что дело в битой закачке. Оплачено было всё окно.

    Поэтому: медиа проверяется ffprobe'ом СРАЗУ и перекачивается. Это то же
    правило, что «лог не свидетель, свидетель файл», только на шаг дальше —
    свидетель не файл, а ЧИТАЕМЫЙ файл."""
    media = os.path.splitext(dest)[1].lower() in (".mp4", ".mov", ".webm", ".mp3", ".wav", ".m4a")
    last = ""
    for k in range(tries):
        subprocess.run(["curl", "-sL", "--retry", "2", "-A", UA, "-o", dest, url], check=True)
        if os.path.getsize(dest) < 1024:
            last = "пустая закачка"
        elif media:
            p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", dest], capture_output=True, text=True)
            if (p.stdout or "").strip():
                return dest
            last = "файл не читается ffprobe (обрыв закачки): " + (p.stderr or "").strip()[:90]
        else:
            return dest
        if k + 1 < tries:
            time.sleep(2.0 * (k + 1))
    raise RuntimeError(f"{last} после {tries} попыток: {url}")


def _run_name():
    """Имя прогона = basename рабочей папки. В v12 тут был хардкод hanbit-ua,
    и чужие прогоны складывали файлы в чужую папку."""
    return re.sub(r"[^\w.-]+", "-", os.path.basename(os.getcwd())) or "run"


def cost_log(kind, label, usd, status):
    """Каждая купленная задача — строка в cost_log.jsonl, ВКЛЮЧАЯ упавшие
    (status: ok|fail). В v12 упавшие не писались, и расход занижался."""
    try:
        with open(os.path.join(os.getcwd(), "cost_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": round(time.time(), 1), "kind": kind, "label": label,
                                "usd": round(usd, 4), "status": status}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _check_refs(refs, what):
    """🔴 Больше 9 рефов — ошибка с перечнем, а НЕ молчаливый срез refs[:9]:
    в v12 первым молча выпадал последний реф — композиционный кадр эталона."""
    if refs and len(refs) > 9:
        raise RuntimeError(f"{what}: рефов {len(refs)} > 9 — модель столько не берёт, "
                           f"а резать молча нельзя. Список:\n  " + "\n  ".join(map(str, refs)))
    return refs


def upload(path, kind="image"):
    out = subprocess.run(["curl", "-s", "-H", "Authorization: Bearer " + key(),
                          "-F", f"file=@{path}", "-F", f"uploadPath={kind}/{_run_name()}",
                          "-F", f"fileName={os.path.basename(path)}", UPLOAD],
                         capture_output=True, text=True).stdout
    m = re.search(r'"downloadUrl"\s*:\s*"([^"]+)"', out)
    if not m: raise RuntimeError("аплоад не отдал downloadUrl: " + out[:250])
    return m.group(1).replace("\\/", "/")


# ---------------- SUNO ----------------
def suno(lyrics, style, title, duration=None, model="V5_5", vocal="f", label="song"):
    """Один запрос = 2 дубля. Возвращает список audioUrl."""
    body = {"customMode": True, "instrumental": False, "model": model,
            "prompt": lyrics[:5000], "style": style[:1000], "title": title[:80],
            "vocalGender": vocal, "callBackUrl": CALLBACK}
    if duration and model == "V5_5":
        body["duration"] = int(duration)          # duration живёт ТОЛЬКО на V5_5
    with _submit:
        r = _http("POST", SUNO_CREATE, body); time.sleep(SUBMIT_GAP)
    tid = ((r.get("data") or {}).get("taskId")) or (r.get("data") or {}).get("task_id")
    if not tid: raise RuntimeError(f"Suno не отдал taskId: {json.dumps(r)[:300]}")
    pipe.job_open(tid, "suno", label, eta=pipe.ETA["suno"])
    pipe.spend(0.06, f"Suno {label}")
    t0 = time.time()
    while time.time() - t0 < POLL_TIMEOUT:
        time.sleep(POLL_EVERY)
        d = (_http("GET", f"{SUNO_INFO}?taskId={tid}").get("data")) or {}
        st = d.get("status") or "?"
        pipe.job_state(tid, "generating" if st in ("PENDING", "TEXT_SUCCESS", "FIRST_SUCCESS") else st.lower())
        if st == "SUCCESS":
            takes = ((d.get("response") or {}).get("sunoData")) or []
            urls = [(t.get("audioUrl") or t.get("sourceAudioUrl"), t.get("duration"), t.get("id"))
                    for t in takes]
            urls = [(u, dur, aid) for u, dur, aid in urls if u]
            pipe.job_state(tid, "success")
            return urls
        if st in ("CREATE_TASK_FAILED", "GENERATE_AUDIO_FAILED", "SENSITIVE_WORD_ERROR"):
            pipe.job_state(tid, "fail")
            raise RuntimeError(f"Suno упал: {st} — {d.get('errorMessage')}")
    pipe.job_state(tid, "fail")
    raise RuntimeError("Suno таймаут")


# ---------------- JOBS (Seedance / nano-banana) ----------------
def _job(model, inp, label, kind, eta):
    with _submit:
        r = _http("POST", JOBS_CREATE, {"model": model, "input": inp, "callBackUrl": CALLBACK})
        time.sleep(SUBMIT_GAP)
    tid = ((r.get("data") or {}).get("taskId")) or (r.get("data") or {}).get("task_id")
    if not tid: raise RuntimeError(f"{model} не отдал taskId: {json.dumps(r)[:300]}")
    pipe.job_open(tid, kind, label, eta=eta)
    t0 = time.time()
    while time.time() - t0 < POLL_TIMEOUT:
        time.sleep(POLL_EVERY)
        d = (_http("GET", f"{JOBS_INFO}?taskId={tid}").get("data")) or {}
        state = (d.get("state") or d.get("status") or "").lower()
        pipe.job_state(tid, state or "generating")
        if state in ("success", "completed"):
            res = d.get("resultJson") or d.get("response") or {}
            if isinstance(res, str): res = json.loads(res)      # 🔴 приходит строкой
            urls = res.get("resultUrls") or res.get("result_urls") or []
            url = urls[0] if urls else (res.get("videoUrl") or res.get("imageUrl") or res.get("url"))
            if not url: raise RuntimeError(f"{model} без url: {json.dumps(res)[:250]}")
            pipe.job_state(tid, "success")
            return url
        if state in ("fail", "failed", "error"):
            pipe.job_state(tid, "fail")
            raise RuntimeError(f"{model} упал: failCode={d.get('failCode')} {d.get('failMsg')}")
    pipe.job_state(tid, "fail")
    raise RuntimeError(f"{model} таймаут")



def _as_urls(refs, kind="image"):
    """🔴 Рефы уходят в модель ТОЛЬКО как URL. Живой прогон 2026-08-20: передали
    локальный путь chars/hero_before.png → nano-banana ответил 422 «media file
    unavailable», и паспорта героя сгенерились без привязки лица — тремя разными
    людьми. Локальный файл заливаем сами, ссылку пропускаем как есть."""
    out = []
    for r in (refs or []):
        out.append(upload(r, kind) if (r and not str(r).startswith("http") and os.path.exists(r)) else r)
    return out

def image(prompt, label, refs=None, aspect="9:16", res="1K"):
    inp = {"prompt": prompt, "image_input": _as_urls(_check_refs(refs, f"кадр {label}")),
           "aspect_ratio": aspect, "resolution": res, "output_format": "png"}
    try:
        with _slots:
            u = _job("nano-banana-2", inp, label, "nano", pipe.ETA["nano"])
    except Exception:
        cost_log("nano", label, pipe.COST["nano"], "fail")
        raise
    pipe.spend(pipe.COST["nano"], f"кадр {label}")
    cost_log("nano", label, pipe.COST["nano"], "ok")
    return u


def video(prompt, first_frame_url, seconds, label, refs=None, res="480p", ref_videos=None):
    """res: 480p (нативно 496x864) или 720p. 🔴 720p ОБЯЗАТЕЛЕН там, где продукт
    в руке или крупным планом — на 480p упаковка и надписи расплываются в кашу.
    Цена: 480p $0.019/с, 720p $0.041/с — в 2.2 раза дороже, поэтому точечно."""
    inp = {"prompt": prompt, "resolution": res, "aspect_ratio": "9:16",
           "duration": int(seconds), "generate_audio": False}
    # 🔴 Все входы живут в ОДНОМ вызове и НЕ исключают друг друга. Было `elif` —
    # при заданном first_frame_url паспорта персонажей молча не уходили вовсе,
    # и на видео-референсе модель взяла людей из чужого ролика (замер 2026-08-21).
    # 🔴 ПЕРВЫЙ КАДР ТОЖЕ ЗАЛИВАЕТСЯ. Замер 23.08.2026: сюда уходил локальный
    # путь ("keys_full/001.png"), API отвечал 400 «The parameter content[0].
    # image_url specified in the request are not valid» и клип не создавался.
    # Это тот же класс бага, что скилл уже ловил на kie.image — там reference
    # молча не применялся, здесь падает весь вызов. Заливаем через ту же
    # функцию, что и остальные входы.
    if first_frame_url:
        inp["first_frame_url"] = _as_urls([first_frame_url])[0]
    if refs:            inp["reference_image_urls"] = _as_urls(_check_refs(refs, f"клип {label}"))
    # 🔴 ВИДЕО-РЕФЕРЕНС: отрезок эталона этого же кадра. Seedance 2 mini умеет
    # брать движение и ход камеры из него — то, что промптом ловится ~в половине
    # случаев. Совместим с first_frame_url: старт наш, движение — эталона.
    if ref_videos:      inp["reference_video_urls"] = _as_urls(list(ref_videos), kind="video")
    usd = pipe.COST["seedance_per_sec_" + res] * int(seconds)
    try:
        with _slots:
            u = _job("bytedance/seedance-2-mini", inp, label, "seedance", pipe.ETA["seedance"])
    except Exception:
        cost_log("seedance", f"{label} {seconds}s {res}", usd, "fail")
        raise
    pipe.spend(usd, f"клип {label} {seconds}s {res}")
    cost_log("seedance", f"{label} {seconds}s {res}", usd, "ok")
    return u


def suno_extend(audio_id, lyrics, style, title, continue_at, model="V5_5", label="продолжение", duration=None):
    """🔴 duration У ЭНДПОИНТА /generate/extend НЕТ. Проверено по спеке kie.ai
    22.08.2026: параметр есть только у /generate и /generate/upload-cover.
    Аргумент принимается ради обратной совместимости и НЕ отправляется: длину
    продолжения задаёт ДЛИНА ТЕКСТА и больше ничто. Следствие для закона нуля:
    единственный рычаг длины песни — количество слогов."""
    """🔴 Продолжение ТОГО ЖЕ трека — одна мелодия, один голос, шва нет.
    Так делаются длинные песни: Suno роняет текст выше ~310 слов, но продолжение
    начинает новый бюджет, музыкально приклеиваясь к оригиналу.
    Контракт замерен: audioId + defaultParamFlag + continueAt обязательны."""
    body = {"audioId": audio_id, "defaultParamFlag": True, "model": model,
            "prompt": lyrics[:5000], "style": style[:1000], "title": title[:80],
            "continueAt": int(continue_at), "callBackUrl": CALLBACK}
    # 🔴 без duration extend не поспішає і викидає рядки: кусок на 165 складів
    # розтягнувся на 88с замість 52 і втратив чотири рядки. duration живе на V5_5.
    if duration and model == "V5_5": body["duration"] = int(duration)
    with _submit:
        r = _http("POST", BASE + "/api/v1/generate/extend", body); time.sleep(SUBMIT_GAP)
    tid = ((r.get("data") or {}).get("taskId"))
    if not tid: raise RuntimeError(f"extend не отдал taskId: {json.dumps(r)[:300]}")
    pipe.job_open(tid, "suno", label, eta=pipe.ETA["suno"])
    pipe.spend(0.06, f"Suno extend {label}")
    t0 = time.time()
    while time.time() - t0 < POLL_TIMEOUT:
        time.sleep(POLL_EVERY)
        d = (_http("GET", f"{SUNO_INFO}?taskId={tid}").get("data")) or {}
        st = d.get("status") or "?"
        pipe.job_state(tid, "generating" if st in ("PENDING","TEXT_SUCCESS","FIRST_SUCCESS") else st.lower())
        if st == "SUCCESS":
            takes = ((d.get("response") or {}).get("sunoData")) or []
            out = [(t.get("audioUrl") or t.get("sourceAudioUrl"), t.get("duration"), t.get("id"))
                   for t in takes]
            pipe.job_state(tid, "success")
            return [(u, dur, aid) for u, dur, aid in out if u]
        if st in ("CREATE_TASK_FAILED","GENERATE_AUDIO_FAILED","SENSITIVE_WORD_ERROR"):
            pipe.job_state(tid, "fail"); raise RuntimeError(f"extend упал: {st} — {d.get('errorMessage')}")
    pipe.job_state(tid, "fail"); raise RuntimeError("extend таймаут")
