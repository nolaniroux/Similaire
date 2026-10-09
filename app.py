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
@lru_cache(maxsize=1024)
def deezer_find(query):
    try:
        data = HTTP.get("https://api.deezer.com/search", params={"q": query, "limit": 1}, timeout=10).json()
        return (data.get("data") or [None])[0]
    except Exception:
        return None


@lru_cache(maxsize=1024)
def deezer_genre(album_id):
    try:
        album = HTTP.get(f"https://api.deezer.com/album/{album_id}", timeout=10).json()
        return (album.get("genres", {}).get("data") or [{}])[0].get("name", "")
    except Exception:
        return ""


# ---------- Analyse audio ----------
@lru_cache(maxsize=512)
def analyze(preview_url):
    """Retourne (bpm, pitch_class, mode) ou None."""
    try:
        import librosa

        audio = HTTP.get(preview_url, timeout=15).content
        y, sr = librosa.load(io.BytesIO(audio), sr=22050, mono=True, duration=30)
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
    for c in lastfm_similar(seed["artist"]["name"], seed["title"]):
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

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = [r for r in pool.map(process, cands) if r]
    results.sort(key=lambda r: -r["score"])
    return jsonify(seed=seed_card, results=results[:8])


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)), debug=True)
