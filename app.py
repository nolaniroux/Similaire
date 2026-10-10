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
DISCOGS_TOKEN = os.environ.get("DISCOGS_TOKEN", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
HTTP = requests.Session()
HTTP.headers["User-Agent"] = "Similaire/1.0"

NOTES = ["Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ---------- Deezer ----------
_cache = {}
_disc = {}


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


# ---------- MusicBrainz + ListenBrainz (artistes similaires, filtre de popularité) ----------
def mb_artist_id(name):
    key = "mb:" + name
    if key in _cache:
        return _cache[key]
    try:
        r = HTTP.get("https://musicbrainz.org/ws/2/artist/",
                     params={"query": f'artist:"{name}"', "fmt": "json", "limit": 1}, timeout=10).json()
        a = (r.get("artists") or [None])[0]
        if a and int(a.get("score", 0)) >= 80:
            _cache[key] = a["id"]
            return a["id"]
    except Exception:
        print("MUSICBRAINZ ECHEC:", traceback.format_exc(), flush=True)
    return None


def lb_similar(artist, niche):
    """Artistes similaires selon ListenBrainz ; le mode niche demande des enregistrements moins populaires."""
    mbid = mb_artist_id(artist)
    if not mbid:
        print("LISTENBRAINZ : artiste introuvable sur MusicBrainz:", artist, flush=True)
        return []
    mode, lo, hi = [("easy", 0, 100), ("medium", 0, 60), ("hard", 0, 35)][niche]
    try:
        r = HTTP.get(f"https://api.listenbrainz.org/1/lb-radio/artist/{mbid}",
                     params={"mode": mode, "max_similar_artists": 30, "max_recordings_per_artist": 2,
                             "pop_begin": lo, "pop_end": hi}, timeout=25).json()
    except Exception:
        print("LISTENBRAINZ ECHEC:", traceback.format_exc(), flush=True)
        return []
    items = []

    def walk(x):  # la réponse est un dict {artiste: [enregistrements]} : on la parcourt sans supposer la forme exacte
        if isinstance(x, list):
            for i in x:
                if isinstance(i, dict) and "recording_mbid" in i:
                    items.append(i)
                else:
                    walk(i)
        elif isinstance(x, dict):
            for k, v in x.items():
                if k != mbid:
                    walk(v)

    walk(r)
    firsts, seen = [], set()
    for i in items:
        k = norm(i.get("similar_artist_name", ""))
        if k and k not in seen and i.get("similar_artist_mbid") != mbid:
            seen.add(k)
            firsts.append(i)
    names = {}
    try:  # noms des morceaux (si l'API ne les donne pas, on prendra le titre phare de l'artiste sur Deezer)
        m = HTTP.post("https://api.listenbrainz.org/1/metadata/recording/",
                      json={"recording_mbids": [i["recording_mbid"] for i in firsts], "inc": "artist"}, timeout=25).json()
        for k, v in m.items():
            names[k] = ((v or {}).get("recording") or {}).get("name") or ""
    except Exception:
        print("LISTENBRAINZ METADATA ECHEC", flush=True)
    n = len(firsts)
    print(f"LISTENBRAINZ: {n} artistes similaires (mode {mode})", flush=True)
    return [{"name": names.get(i["recording_mbid"], ""), "artist": {"name": i.get("similar_artist_name", ""), "mbid": i.get("similar_artist_mbid")},
             "match": round((0.7 if niche else 0.45) - 0.3 * idx / max(1, n - 1), 3), "playcount": 0}
            for idx, i in enumerate(firsts)]


def fetch_pools(seed, niche):
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(get_candidates, seed, niche)
        f2 = ex.submit(lb_similar, seed["artist"]["name"], niche)
        return f1.result(), f2.result()


def merge_candidates(lfm, lb):
    """Fusionne Last.fm et ListenBrainz : un artiste proposé par les deux sources remonte."""
    by = {}
    for c in lfm:
        c["src"] = ["Last.fm"]
        by[norm(c["artist"]["name"])] = c
    for c in lb:
        k = norm(c["artist"]["name"])
        if not k:
            continue
        if k in by:
            by[k]["src"].append("ListenBrainz")
            by[k]["match"] = min(1.0, float(by[k].get("match", 0)) + 0.15)
        else:
            c["src"] = ["ListenBrainz"]
            by[k] = c
    return sorted(by.values(), key=lambda c: -float(c.get("match", 0)))


def lb_popularity(mbids):
    """Nombre d'auditeurs distincts par artiste d'après ListenBrainz (une seule requête groupée)."""
    mbids = list(dict.fromkeys(m for m in mbids if m))
    out = {}
    if not mbids:
        return out
    try:
        r = HTTP.post("https://api.listenbrainz.org/1/popularity/artist", json={"artist_mbids": mbids}, timeout=20).json()
        items = r if isinstance(r, list) else (r.get("payload") or r.get("artists") or [])
        for i in items:
            if isinstance(i, dict) and i.get("artist_mbid") and i.get("total_user_count") is not None:
                out[i["artist_mbid"]] = int(i["total_user_count"])
    except Exception:
        print("LISTENBRAINZ POPULARITE ECHEC:", traceback.format_exc(), flush=True)
    print(f"LISTENBRAINZ POPULARITE: {len(out)}/{len(mbids)} artistes trouvés", flush=True)
    return out


def rarity(r):
    """0 = énorme succès ... 1 = très confidentiel. Moyenne des indices disponibles (Deezer, ListenBrainz)."""
    vals = []
    if r.get("rank", 0) > 0:
        vals.append(1 - min(1.0, r["rank"] / 1_000_000))
    if r.get("lb_users") is not None:
        vals.append(1 - min(1.0, math.log10(r["lb_users"] + 1) / 6))
    return sum(vals) / len(vals) if vals else 0.5


def deezer_top(artist, niche=0):
    """Un morceau d'un artiste sur Deezer : le plus connu, ou en mode niche le moins populaire de ses 10 titres phares."""
    d = deezer_get("https://api.deezer.com/search/artist", {"q": artist, "limit": 1})
    a = ((d or {}).get("data") or [None])[0]
    if not a or norm(a["name"]) != norm(artist):
        return None
    t = deezer_get(f"https://api.deezer.com/artist/{a['id']}/top", {"limit": 10})
    items = (t or {}).get("data") or []
    if not items:
        return None
    return items[0] if not niche else min(items, key=lambda x: x.get("rank", 0))


# ---------- Claude (optionnel) : accords et instruments ----------
INSTRUMENTS = ["guitare saturée", "guitare claire", "guitare acoustique", "basse", "basse slap", "batterie",
               "piano", "synthétiseur", "orgue", "violon", "cordes", "saxophone", "trompette", "cuivres",
               "flûte", "percussions", "boîte à rythmes", "voix chantée", "voix rappée", "voix criée", "chœurs"]
# instruments présents dans presque tous les morceaux rock/pop : ils comptent moins dans la comparaison
COMMON = {norm(x) for x in ["basse", "batterie", "voix chantée", "chœurs"]}


# mots de genre trop généraux : ils comptent moins que « funk », « post-punk », « trap »...
GENERIC = {"rock", "pop", "music", "musique", "alternatif", "alternative", "moderne", "contemporain"}


def genre_tokens(genres):
    return {t for g in genres for t in re.findall(r"[a-zà-ÿ0-9]+", str(g).lower()) if len(t) > 1}


def genre_sim(a, b):
    """Similarité pondérée entre deux ensembles de mots de sous-genres."""
    if not a or not b:
        return 0.0
    w = lambda x: 0.4 if x in GENERIC else 1.0
    return sum(w(x) for x in a & b) / sum(w(x) for x in a | b)


def inst_sim(a, b):
    """Similarité pondérée (type Jaccard) entre deux ensembles d'instruments."""
    if not a or not b:
        return 0.0
    w = lambda x: 0.4 if x in COMMON else 1.0
    return sum(w(x) for x in a & b) / sum(w(x) for x in a | b)


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
        "et les instruments principaux audibles, plus 2 à 3 sous-genres précis en minuscules "
        "(ex : funk rock, indie pop, post-punk, trap). Pour les instruments, utilise uniquement "
        f"des mots de cette liste : {', '.join(INSTRUMENTS)}. Garde les 4 à 6 éléments les plus caractéristiques "
        f"du son (la texture de la guitare compte : saturée, claire ou acoustique).\n{lines}\n"
        'Réponds avec un tableau JSON, un objet par chanson, dans le même ordre : '
        '[{"i":0,"accords":"...","instruments":["..."],"genres":["..."]}]'
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
        out = {int(o["i"]): {"accords": o.get("accords", ""), "instruments": o.get("instruments", []), "genres": o.get("genres", [])} for o in arr}
        _desc_cache[key] = out
        return out, ""
    except Exception:
        print("IA ECHEC:", traceback.format_exc(), flush=True)
        return None, "erreur de l'IA (voir les Logs Render)."


def card(t):
    return {
        "title": t["title"], "artist": t["artist"]["name"],
        "cover": t["album"].get("cover_medium", ""), "link": t.get("link", ""),
        "preview": t.get("preview", ""), "genre": deezer_genre(t["album"]["id"]), "rank": t.get("rank", 0),
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
    for c in rank_pool(merge_candidates(*fetch_pools(seed, niche)), niche):
        a = c["artist"]["name"]
        if a.lower() not in seen:  # un seul titre par artiste
            seen.add(a.lower())
            cands.append(c)
        if len(cands) >= (45 if niche else 30):
            break

    def process(c):
        art, name = c["artist"]["name"], clean_title(c.get("name") or "")
        src = c.get("src", [])
        t = (deezer_find(f'artist:"{art}" track:"{name}"') or deezer_find(f"{art} {name}")) if name else None
        if not t and "ListenBrainz" in src:
            t = deezer_top(art, niche)  # pas de titre connu : on prend son morceau le plus connu
        if not t:
            print("NON TROUVE SUR DEEZER:", art, "-", name, flush=True)
            return None
        da = t["artist"]["name"]
        if norm(art) not in norm(da) and norm(da) not in norm(art):
            print("ARTISTE DIFFERENT:", art, "vs", da, flush=True)
            return None  # non vérifié : on l'écarte
        res = card(t)
        if src == ["ListenBrainz"] and res["genre"] and seed_card["genre"] and res["genre"] != seed_card["genre"]:
            print("LISTENBRAINZ ECARTE (genre different):", art, "-", res["genre"], flush=True)
            return None  # une seule source et un autre genre : trop risqué
        score = max(0.05, 0.9 * float(c.get("match", 0)) - 0.05 * niche * popularity(c)
                    - 0.25 * niche * min(1.0, t.get("rank", 0) / 1_000_000))
        why = []
        if "Last.fm" in src:
            why.append("écoutes communes (Last.fm)")
        if "ListenBrainz" in src:
            why.append("artiste similaire (ListenBrainz)")
        if len(src) > 1:
            score += 0.08
            why.append("confirmée par 2 sources")
        res["sources"] = src
        res["artist_mbid"] = c["artist"].get("mbid") or ""
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
    if niche:  # ListenBrainz : nombre d'auditeurs de chaque artiste
        users = lb_popularity([r["artist_mbid"] for r in results if r.get("artist_mbid")])
        for r in results:
            if users.get(r.get("artist_mbid")) is not None:
                r["lb_users"] = users[r["artist_mbid"]]
    for r in results:
        r["rarity"] = round(rarity(r), 2)
        r["rare"] = r["rarity"] >= 0.55
    if niche:  # vrai filtre de popularité (Deezer + ListenBrainz)
        cap = [0, 0.35, 0.55][niche]
        keep = [r for r in results if r["rarity"] >= cap]
        print(f"NICHE {niche}: {len(keep)} titres assez confidentiels sur {len(results)}", flush=True)
        results = keep if len(keep) >= 8 else sorted(results, key=lambda r: -r["rarity"])[:12]
    results.sort(key=lambda r: -r["score"])
    results = results[:30]
    info, err = describe_many([(seed["title"], seed["artist"]["name"])] + [(r["title"], r["artist"]) for r in results])
    notice, grouped = "", False
    if info:
        seed_card.update(info.get(0, {}))
        ref = {norm(x) for x in seed_card.get("instruments", []) if x}
        ref_distinct = ref - COMMON
        sg = genre_tokens(seed_card.get("genres", []))
        for i, r in enumerate(results, 1):
            r.update(info.get(i, {}))
            got = {norm(x) for x in r.get("instruments", []) if x}
            sim = inst_sim(ref, got)
            # mêmes instruments = forte similarité ET au moins un instrument distinctif en commun
            r["same"] = bool(ref) and sim >= 0.5 and (bool(ref_distinct & got) or not ref_distinct)
            if r["same"]:
                r["score"] = min(100, r["score"] + round(10 * sim))
            gs = genre_sim(sg, genre_tokens(r.get("genres", [])))
            r["same_genre"] = gs >= 0.5
            if r["same_genre"]:
                r["score"] = min(100, r["score"] + round(10 * gs))
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


@app.route("/api/discogs")
def discogs_api():
    """Styles, année et nombre de collectionneurs d'un morceau d'après Discogs (appelé par la page)."""
    if not DISCOGS_TOKEN:
        return jsonify(error="DISCOGS_TOKEN non configuré")
    artist = request.args.get("artist", "")[:100]
    title = clean_title(request.args.get("title", ""))[:150]
    key = (artist, title)
    if key in _disc:
        return jsonify(_disc[key])
    try:
        r = HTTP.get("https://api.discogs.com/database/search",
                     params={"artist": artist, "track": title, "type": "release", "per_page": 5},
                     headers={"Authorization": f"Discogs token={DISCOGS_TOKEN}"}, timeout=15).json()
        res = r.get("results") or []
        if not res:
            return jsonify(error="introuvable sur Discogs")
        styles, genres = [], []
        for x in res[:3]:
            styles += x.get("style", []) or []
            genres += x.get("genre", []) or []
        uniq = lambda a: list(dict.fromkeys(a))
        out = {"styles": uniq(styles)[:6], "genres": uniq(genres)[:3], "year": res[0].get("year"),
               "have": (res[0].get("community") or {}).get("have", 0)}
        _disc[key] = out
        return jsonify(out)
    except Exception:
        print("DISCOGS ECHEC:", traceback.format_exc(), flush=True)
        return jsonify(error="erreur Discogs")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)), debug=True)
