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
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import requests
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")
LASTFM_KEY = os.environ.get("LASTFM_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
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


def key_name(pc, mode):
    return f"{NOTES[pc]} {'majeur' if mode == 'maj' else 'mineur'}"


def tempo_sim(a, b):
    d = abs(math.log2(a / b)) % 1  # insensible à l'erreur x2 / ÷2 du détecteur
    d = min(d, 1 - d)
    return max(0.0, 1 - d / 0.3)


def key_sim(k1, k2):
    def fifths(pc, mode):
        major = pc if mode == "maj" else (pc + 3) % 12  # on ramène au relatif majeur
        return (major * 7) % 12

    d = abs(fifths(*k1) - fifths(*k2))
    d = min(d, 12 - d)
    return max(0.0, 1 - d / 4)


# ---------- Last.fm ----------
def lastfm_similar(artist, title):
    r = HTTP.get(
        "https://ws.audioscrobbler.com/2.0/",
        params={"method": "track.getsimilar", "artist": artist, "track": title, "limit": 30,
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


def get_candidates(seed):
    artist, title = seed["artist"]["name"], clean_title(seed["title"])
    found = lastfm_similar(artist, title)
    if found:
        return found
    # Plan B : artistes similaires, puis leur titre le plus connu
    arts = lastfm_call("artist.getsimilar", artist=artist, limit=14).get("similarartists", {}).get("artist", [])

    def top(a):
        try:
            t = lastfm_call("artist.gettoptracks", artist=a["name"], limit=1).get("toptracks", {}).get("track", [])
            return {"name": t[0]["name"], "artist": {"name": a["name"]}, "match": float(a.get("match", 0)) * 0.8} if t else None
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        return [r for r in pool.map(top, arts) if r]


# ---------- Claude (optionnel) : accords et instruments ----------
@lru_cache(maxsize=256)
def describe(title, artist):
    if not ANTHROPIC_KEY:
        return {}
    try:
        r = HTTP.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
            json={"model": "claude-sonnet-5-5", "max_tokens": 400, "messages": [{"role": "user", "content":
                  f'Chanson : "{title}" de {artist}. Réponds uniquement en JSON : '
                  '{"accords":"progression typique","instruments":["..."]}'}]},
            timeout=30,
        ).json()
        text = r["content"][0]["text"]
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except Exception:
        return {}


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
    if not q:
        return jsonify(error="Entre un titre."), 400
    if not LASTFM_KEY:
        return jsonify(error="Clé LASTFM_API_KEY manquante sur le serveur."), 500

    seed = deezer_find(q)
    if not seed:
        return jsonify(error="Chanson introuvable. Essaie « titre artiste »."), 404
    seed_card = card(seed)
    seed_an = analyze(seed["preview"]) if seed.get("preview") else None
    extra = describe(seed["title"], seed["artist"]["name"])
    seed_card.update(extra)
    if seed_an:
        seed_card["bpm"], seed_card["key"] = seed_an[0], key_name(seed_an[1], seed_an[2])

    seen = {seed["artist"]["name"].lower()}
    cands = []
    for c in get_candidates(seed):
        a = c["artist"]["name"]
        if a.lower() not in seen:  # un seul titre par artiste
            seen.add(a.lower())
            cands.append(c)
        if len(cands) >= 14:
            break

    def process(c):
        t = deezer_find(f'artist:"{c["artist"]["name"]}" track:"{c["name"]}"')
        if not t or c["artist"]["name"].lower() not in t["artist"]["name"].lower():
            return None  # non vérifié sur Deezer : on l'écarte
        res = card(t)
        an = analyze(t["preview"]) if t.get("preview") else None
        score, why = 0.6 * float(c.get("match", 0)), ["écoutes communes sur Last.fm"]
        if an and seed_an:
            ts, ks = tempo_sim(an[0], seed_an[0]), key_sim(an[1:], seed_an[1:])
            score += 0.25 * ts + 0.15 * ks
            res["bpm"], res["key"] = an[0], key_name(an[1], an[2])
            if ts > 0.7: why.append(f"tempo proche ({an[0]} BPM)")
            if ks > 0.7: why.append("tonalité compatible")
        else:
            score += 0.2
        if res["genre"] and res["genre"] == seed_card["genre"]:
            score += 0.05
            why.append(f"même genre ({res['genre']})")
        res["score"], res["why"] = round(score * 100), ", ".join(why)
        return res

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = [r for r in pool.map(process, cands) if r]
    print(f"CANDIDATS: {len(cands)} -> VERIFIES: {len(results)}", flush=True)
    results.sort(key=lambda r: -r["score"])
    return jsonify(seed=seed_card, results=results[:8])


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)), debug=True)
