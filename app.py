"""Similaire : recommandations musicales basées sur de vraies données.

- Deezer  : vérifie que le morceau existe + extrait audio de 30 s + genre
- Last.fm : vraies chansons similaires (écoutes de millions d'utilisateurs)
- librosa : mesure le tempo (BPM) et la tonalité sur l'extrait audio
- Claude (optionnel) : décrit accords et instruments si ANTHROPIC_API_KEY est défini
"""
import io
import json
import math
import os
import re
import threading
import time
from urllib.parse import urlparse
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import requests
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")
LASTFM_KEY = os.environ.get("LASTFM_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
HTTP = requests.Session()
HTTP.headers["User-Agent"] = "Similaire/1.0"

NOTES = ["Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ---------- Deezer ----------
_cache = {}


def deezer_get(url, params=None):
    """Appelle Deezer avec nouvel essai si la limite de requêtes est atteinte."""
    for i in range(3):
        try:
            d = HTTP.get(url, params=params, timeout=10).json()
        except Exception:
            d = {"error": "reseau"}
        if "error" not in d:
            return d
        print("DEEZER ERREUR:", d["error"], flush=True)
        time.sleep(1 + i)
    return None


def deezer_find(query):
    if query in _cache:
        return _cache[query]
    d = deezer_get("https://api.deezer.com/search", {"q": query, "limit": 1})
    t = ((d or {}).get("data") or [None])[0]
    if t:
        _cache[query] = t
    return t


def deezer_find_seed(q):
    """Choisit, parmi les 8 premiers résultats, le plus populaire qui correspond bien à la recherche
    (évite de tomber sur une parodie ou une reprise obscure)."""
    d = deezer_get("https://api.deezer.com/search", {"q": q, "limit": 8})
    items = (d or {}).get("data") or []
    toks = [norm(w) for w in q.split() if len(norm(w)) > 2]
    good = [t for t in items if all(k in norm(t["title"] + t["artist"]["name"]) for k in toks)]
    pool = good or items
    return max(pool, key=lambda t: t.get("rank", 0)) if pool else None


def deezer_genre(album_id):
    key = f"album:{album_id}"
    if key not in _cache:
        d = deezer_get(f"https://api.deezer.com/album/{album_id}") or {}
        _cache[key] = (d.get("genres", {}).get("data") or [{}])[0].get("name", "")
    return _cache[key]


# ---------- Analyse audio ----------
@lru_cache(maxsize=512)
def analyze(preview_url):
    """Retourne (bpm, pitch_class, mode) ou None."""
    try:
        import librosa

        import miniaudio

        audio = HTTP.get(preview_url, timeout=15).content
        dec = miniaudio.decode(audio, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1, sample_rate=22050)
        y = np.array(dec.samples, dtype=np.float32) / 32768.0
        sr = 22050
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        while bpm < 80:  # on ramène tout dans 80-160 BPM : évite les erreurs x2 / ÷2 du détecteur
            bpm *= 2
        while bpm >= 160:
            bpm /= 2
        chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
        best = (-2, 0, "maj")
        for pc in range(12):
            rolled = np.roll(chroma, -pc)
            for mode, prof in (("maj", MAJ), ("min", MIN)):
                c = np.corrcoef(rolled, prof)[0, 1]
                if c > best[0]:
                    best = (c, pc, mode)
        return round(bpm), best[1], best[2]
    except Exception:
        print("ANALYSE ECHEC:", traceback.format_exc(), flush=True)
        return None


def norm(x):
    return re.sub(r"[^a-z0-9]", "", x.lower())


def key_name(pc, mode):
    return f"{NOTES[pc]} {'majeur' if mode == 'maj' else 'mineur'}"


def tempo_sim(a, b):
    d = abs(math.log2(a / b))  # tempos déjà ramenés dans la même plage
    return max(0.0, 1 - d / 0.3)


def key_sim(k1, k2):
    def fifths(pc, mode):
        major = pc if mode == "maj" else (pc + 3) % 12  # on ramène au relatif majeur
        return (major * 7) % 12

    d = abs(fifths(*k1) - fifths(*k2))
    d = min(d, 12 - d)
    return max(0.0, 1 - d / 4)


# ---------- Last.fm ----------
def lastfm_similar(artist, title, limit=30):
    r = HTTP.get(
        "https://ws.audioscrobbler.com/2.0/",
        params={"method": "track.getsimilar", "artist": artist, "track": title, "limit": limit,
                "autocorrect": 1, "api_key": LASTFM_KEY, "format": "json"},
        timeout=15,
    ).json()
    return r.get("similartracks", {}).get("track", [])


def clean_title(t):
    t = re.sub(r"^\d{6,8}\s+", "", t)  # préfixe de date
    t = re.sub(r"\s*[\(\[].*?[\)\]]", "", t)  # (Live), [Remastered]...
    t = re.sub(r"\s+-\s+(remaster|live|radio|single|version|acoustic).*$", "", t, flags=re.I)
    return t.strip()


def lastfm_call(method, **params):
    return HTTP.get("https://ws.audioscrobbler.com/2.0/",
                    params={"method": method, "api_key": LASTFM_KEY, "format": "json", "autocorrect": 1, **params},
                    timeout=15).json()


def popularity(c):
    """0 = quasi inconnue ... 5 = énorme (d'après le nombre d'écoutes Last.fm)."""
    return max(0.0, math.log10(int(c.get("playcount") or 0) + 1) - 3)


def rank_pool(pool, niche):
    """Mode découverte : on fait remonter les titres moins écoutés (mais encore proches)."""
    if not niche:
        return pool
    pool = [c for c in pool if float(c.get("match", 0)) >= 0.1]
    return sorted(pool, key=lambda c: float(c.get("match", 0)) - 0.08 * niche * popularity(c), reverse=True)


def get_candidates(seed, niche=0):
    artist, title = seed["artist"]["name"], clean_title(seed["title"])
    found = lastfm_similar(artist, title, 100 if niche else 30)
    if found:
        return found
    # Plan B : artistes similaires, puis leur titre le plus connu
    arts = lastfm_call("artist.getsimilar", artist=artist, limit=30).get("similarartists", {}).get("artist", [])

    def top(a):
        try:
            t = lastfm_call("artist.gettoptracks", artist=a["name"], limit=1).get("toptracks", {}).get("track", [])
            return {"name": t[0]["name"], "artist": {"name": a["name"]}, "match": float(a.get("match", 0)) * 0.8} if t else None
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        return [r for r in pool.map(top, arts) if r]


# ---------- Claude (optionnel) : accords et instruments ----------
INSTRUMENTS = ["guitare", "piano", "batterie", "basse", "synthétiseur", "violon",
               "saxophone", "trompette", "flûte", "orgue", "percussions", "voix"]
_desc_cache = {}


def describe_many(items):
    """Une seule requête Gemini pour toutes les chansons.
    items = [(titre, artiste), ...] -> ({index: {accords, instruments}}, message_erreur)"""
    if not GEMINI_KEY:
        return None, "clé GEMINI_API_KEY non configurée sur Render."
    key = tuple(items)
    if key in _desc_cache:
        return _desc_cache[key], ""
    lines = "\n".join(f"{i}. {t} — {a}" for i, (t, a) in enumerate(items))
    prompt = (
        "Pour chacune de ces chansons, donne en français la progression d'accords typique "
        "et les instruments principaux audibles. Pour les instruments, utilise uniquement "
        f"des mots de cette liste : {', '.join(INSTRUMENTS)}.\n{lines}\n"
        'Réponds avec un tableau JSON, un objet par chanson, dans le même ordre : '
        '[{"i":0,"accords":"...","instruments":["..."]}]'
    )
    try:
        r = HTTP.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            headers={"x-goog-api-key": GEMINI_KEY},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}},
            timeout=60,
        ).json()
        if "candidates" not in r:
            msg = (r.get("error") or {}).get("message", "réponse vide")
            print("IA ECHEC (Gemini):", msg, flush=True)
            return None, "Gemini : " + msg[:150]
        text = r["candidates"][0]["content"]["parts"][0]["text"]
        arr = json.loads(text[text.index("["): text.rindex("]") + 1])
        out = {int(o["i"]): {"accords": o.get("accords", ""), "instruments": o.get("instruments", [])} for o in arr}
        _desc_cache[key] = out
        return out, ""
    except Exception:
        print("IA ECHEC:", traceback.format_exc(), flush=True)
        return None, "erreur de l'IA (voir les Logs Render)."


def card(t):
    return {
        "title": t["title"], "artist": t["artist"]["name"],
        "cover": t["album"].get("cover_medium", ""), "link": t.get("link", ""),
        "preview": t.get("preview", ""), "genre": deezer_genre(t["album"]["id"]),
    }


# ---------- API ----------
@app.route("/api/similar")
def similar():
    q = request.args.get("q", "").strip()
    try:
        niche = min(2, max(0, int(request.args.get("niche", "0"))))  # 0 populaires, 1 découverte, 2 très niche
    except ValueError:
        niche = 0
    if not q:
        return jsonify(error="Entre un titre."), 400
    if not LASTFM_KEY:
        return jsonify(error="Clé LASTFM_API_KEY manquante sur le serveur."), 500

    seed = deezer_find_seed(q)
    if not seed:
        return jsonify(error="Chanson introuvable. Essaie « titre artiste »."), 404
    seed_card = card(seed)
    seed_an = analyze(seed["preview"]) if seed.get("preview") else None
    if seed_an:
        seed_card["bpm"], seed_card["key"] = seed_an[0], key_name(seed_an[1], seed_an[2])
        seed_card["pc"], seed_card["mode"] = seed_an[1], seed_an[2]

    seen = {seed["artist"]["name"].lower()}
    cands = []
    for c in rank_pool(get_candidates(seed, niche), niche):
        a = c["artist"]["name"]
        if a.lower() not in seen:  # un seul titre par artiste
            seen.add(a.lower())
            cands.append(c)
        if len(cands) >= 30:
            break

    def process(c):
        art, name = c["artist"]["name"], clean_title(c["name"])
        t = deezer_find(f'artist:"{art}" track:"{name}"') or deezer_find(f"{art} {name}")
        if not t:
            print("NON TROUVE SUR DEEZER:", art, "-", name, flush=True)
            return None
        da = t["artist"]["name"]
        if norm(art) not in norm(da) and norm(da) not in norm(art):
            print("ARTISTE DIFFERENT:", art, "vs", da, flush=True)
            return None  # non vérifié : on l'écarte
        res = card(t)
        score = max(0.05, 0.9 * float(c.get("match", 0)) - 0.05 * niche * popularity(c))
        why = ["écoutes communes sur Last.fm"]
        plays = int(c.get("playcount") or 0)
        res["plays"] = plays
        if niche and 0 < plays < 500000:
            why.append("peu connue du grand public")
        if res["genre"] and res["genre"] == seed_card["genre"]:
            score += 0.05
            why.append(f"même genre ({res['genre']})")
        res["score"], res["why"] = round(score * 100), ", ".join(why)
        return res

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = [r for r in pool.map(process, cands) if r]
    print(f"CANDIDATS: {len(cands)} -> VERIFIES: {len(results)}", flush=True)
    results.sort(key=lambda r: -r["score"])
    results = results[:30]
    info, err = describe_many([(seed["title"], seed["artist"]["name"])] + [(r["title"], r["artist"]) for r in results])
    notice, grouped = "", False
    if info:
        seed_card.update(info.get(0, {}))
        ref = {norm(x) for x in seed_card.get("instruments", []) if x}
        need = max(1, math.ceil(len(ref) / 2))  # au moins la moitié des instruments de la chanson
        for i, r in enumerate(results, 1):
            r.update(info.get(i, {}))
            com = [x for x in r.get("instruments", []) if any(norm(x) == n or n in norm(x) for n in ref)]
            r["same"] = bool(ref) and len(com) >= need
            if r["same"]:
                r["why"] += ", instruments communs : " + ", ".join(com)
        grouped = bool(ref)
        if grouped:
            same = [r for r in results if r["same"]][:20]
            results = same + [r for r in results if not r["same"]][: 20 - len(same)]
        else:
            notice = "Les instruments de ta chanson n'ont pas pu être identifiés."
    else:
        notice = "Instruments non analysés : " + err
    return jsonify(seed=seed_card, results=results[:20], notice=notice, grouped=grouped)


 
ANALYZE_LOCK = threading.Lock()


@app.route("/api/analyze")
def analyze_api():
    """Analyse un extrait Deezer (appelé un par un par la page)."""
    url = request.args.get("url", "")
    host = urlparse(url).hostname or ""
    if not url.startswith("https://") or not host.endswith(".dzcdn.net"):
        return jsonify(error="url invalide"), 400
    with ANALYZE_LOCK:  # une seule analyse à la fois (mémoire)
        an = analyze(url)
    if not an:
        return jsonify(error="analyse impossible")
    out = {"bpm": an[0], "key": key_name(an[1], an[2])}
    try:
        seed = (int(request.args["pc"]), request.args["mode"])
        out["tempo_sim"] = round(tempo_sim(an[0], float(request.args["bpm"])), 2)
        out["key_sim"] = round(key_sim(an[1:], seed), 2)
    except (KeyError, ValueError):
        pass
    return jsonify(out)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)), debug=True)
