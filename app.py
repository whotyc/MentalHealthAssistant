import datetime
import json
import os
import re
import sqlite3
from urllib.parse import quote

import requests
from flask import Flask, g, jsonify, render_template, request
from flask_jwt_extended import (JWTManager, create_access_token, get_jwt_identity,
                                jwt_required)
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "dev-secret-change-me")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = datetime.timedelta(days=7)
jwt = JWTManager(app)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mental_health.db")
LANGS = {"ru": "Russian", "en": "English", "zh": "Simplified Chinese",
         "de": "German", "fr": "French", "es": "Spanish"}
EMOTIONS = {"happy", "neutral", "sad", "angry", "fearful", "disgusted", "surprised"}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

CRISIS_RE = re.compile(
    r"суицид|самоубийств|покончить с собой|не хочу жить|убить себя|хочу умереть|"
    r"suicid|kill myself|end my life|want to die|don'?t want to live|"
    r"自杀|轻生|不想活|想死|selbstmord|suizid|umbringen|nicht mehr leben|"
    r"me tuer|veux mourir|envie de mourir|quiero morir|matarme|no quiero vivir",
    re.I)

CRISIS = {
    "ru": "Мне очень жаль, что сейчас так тяжело. Спасибо, что написали об этом. Ваша безопасность важнее всего: пожалуйста, прямо сейчас свяжитесь с близким человеком или со службой экстренной помощи вашей страны (в России и ЕС — 112, в США — 988). Вы не одни, помощь существует. Если хотите, я побуду с вами в разговоре.",
    "en": "I'm really sorry you're going through this, and thank you for telling me. Your safety matters most: please contact someone you trust or your local emergency service right now (112 in the EU, 988 in the US). You are not alone, and help is available. I'm here to keep talking if you'd like.",
    "zh": "听到你正经历这么艰难的时刻，我很难过，谢谢你告诉我。你的安全最重要：请立即联系你信任的人或当地的紧急求助服务。你并不孤单，帮助是存在的。如果你愿意，我会继续陪你聊。",
    "de": "Es tut mir sehr leid, dass es dir gerade so schwerfällt, und danke, dass du es mir sagst. Deine Sicherheit ist am wichtigsten: Bitte wende dich jetzt an eine Vertrauensperson oder den Notruf deines Landes (112 in der EU). Du bist nicht allein, es gibt Hilfe. Ich bin gern weiter für dich da.",
    "fr": "Je suis vraiment désolé que ce soit si difficile en ce moment, et merci de m'en parler. Votre sécurité passe avant tout : contactez dès maintenant un proche ou les services d'urgence de votre pays (112 dans l'UE). Vous n'êtes pas seul(e), de l'aide existe. Je reste là pour discuter si vous le souhaitez.",
    "es": "Lamento mucho que estés pasando por esto, y gracias por contármelo. Tu seguridad es lo más importante: por favor contacta ahora con alguien de confianza o con los servicios de emergencia de tu país (112 en la UE). No estás solo/a, existe ayuda. Sigo aquí para hablar si quieres.",
}

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_=None):
    conn = g.pop("db", None)
    if conn:
        conn.close()


def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL, password_hash TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS measurements(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL, emotion TEXT NOT NULL, emotions_json TEXT NOT NULL,
            stress INTEGER NOT NULL, details_json TEXT, insight TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id));
        CREATE TABLE IF NOT EXISTS chats(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id));
        CREATE INDEX IF NOT EXISTS idx_m_user ON measurements(user_id, id);
        CREATE INDEX IF NOT EXISTS idx_c_user ON chats(user_id, id);
        """)


init_db()


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def uid():
    return int(get_jwt_identity())


def make_token(user_id):
    return create_access_token(identity=str(user_id))


# ---------- ИИ ----------
def _chat_completion(url, headers, payload):
    r = requests.post(url, json=payload, headers=headers, timeout=45)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def ask_ai(messages, max_tokens=600):
    """1) любой OpenAI-совместимый API (Groq/OpenRouter/Ollama...), если задан AI_API_KEY;
       2) иначе бесплатный Pollinations без ключа. Возвращает None, если всё недоступно."""
    key = os.getenv("AI_API_KEY")
    if key:
        base = os.getenv("AI_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        try:
            return _chat_completion(
                base + "/chat/completions",
                {"Authorization": "Bearer " + key},
                {"model": os.getenv("AI_MODEL", "llama-3.1-8b-instant"),
                 "messages": messages, "max_tokens": max_tokens, "temperature": 0.7})
        except Exception as e:
            app.logger.warning("AI provider failed: %s", e)
    try:
        return _chat_completion("https://text.pollinations.ai/openai", {},
                                {"model": "openai", "messages": messages, "private": True})
    except Exception as e:
        app.logger.warning("Pollinations failed: %s", e)
    return None


def system_prompt(lang, ctx=""):
    return (
        "You are a warm, empathetic supportive companion ('AI psychologist') inside a mental "
        "wellbeing app. Use evidence-based, practical techniques (CBT, mindfulness, breathing, "
        "grounding) in simple language. Reply in " + LANGS[lang] + ". Keep replies concise "
        "(3-6 sentences) and ask at most one gentle follow-up question. You are not a doctor: "
        "never diagnose or advise medication. If the user mentions self-harm or suicide, respond "
        "with care and urge them to contact local emergency services or a trusted person. " + ctx)

MOOD_QUERY = {"happy": "happy hits", "neutral": "lofi focus", "sad": "comforting acoustic",
              "angry": "calm down relax", "fearful": "calm ambient relax",
              "disgusted": "mood booster", "surprised": "feel good"}


def music_for(emotion, stress):
    q = "relaxing meditation" if stress >= 67 else MOOD_QUERY.get(emotion, "chill")
    out = {"query": q, "playlists": [],
           "search_url": "https://open.spotify.com/search/" + quote(q + " playlist")}
    cid, secret = os.getenv("SPOTIFY_CLIENT_ID"), os.getenv("SPOTIFY_CLIENT_SECRET")
    if cid and secret:
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
            sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=cid, client_secret=secret))
            res = sp.search(q=q, type="playlist", limit=4)
            for i in res["playlists"]["items"]:
                if i:
                    out["playlists"].append({"name": i["name"],
                                             "url": i["external_urls"]["spotify"]})
        except Exception as e:
            app.logger.warning("Spotify failed: %s", e)
    return out


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/register")
def register():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()[:60]
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not name or not email or not password:
        return jsonify(error="fill_fields"), 400
    if not EMAIL_RE.match(email):
        return jsonify(error="bad_email"), 400
    if len(password) < 6:
        return jsonify(error="short_password"), 400
    try:
        cur = db().execute(
            "INSERT INTO users(email,name,password_hash,created_at) VALUES (?,?,?,?)",
            (email, name, generate_password_hash(password), now_iso()))
        db().commit()
    except sqlite3.IntegrityError:
        return jsonify(error="email_exists"), 409
    return jsonify(token=make_token(cur.lastrowid),
                   user={"id": cur.lastrowid, "name": name, "email": email}), 201


@app.post("/api/login")
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    row = db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not row or not check_password_hash(row["password_hash"], data.get("password") or ""):
        return jsonify(error="invalid_credentials"), 401
    return jsonify(token=make_token(row["id"]),
                   user={"id": row["id"], "name": row["name"], "email": row["email"]})


@app.get("/api/me")
@jwt_required()
def me():
    row = db().execute("SELECT id,name,email FROM users WHERE id=?", (uid(),)).fetchone()
    if not row:
        return jsonify(msg="user not found"), 401
    return jsonify(dict(row))

def row_to_item(r):
    return {"id": r["id"], "created_at": r["created_at"], "emotion": r["emotion"],
            "emotions": json.loads(r["emotions_json"]), "stress": r["stress"],
            "details": json.loads(r["details_json"] or "{}"), "insight": r["insight"]}


@app.post("/api/measurements")
@jwt_required()
def add_measurement():
    data = request.get_json(silent=True) or {}
    emotion, stress = data.get("emotion"), data.get("stress")
    emotions, details = data.get("emotions"), data.get("details") or {}
    if emotion not in EMOTIONS or not isinstance(stress, (int, float)) \
            or not isinstance(emotions, dict):
        return jsonify(error="fill_fields"), 400
    stress = int(max(0, min(100, stress)))
    cur = db().execute(
        "INSERT INTO measurements(user_id,created_at,emotion,emotions_json,stress,details_json)"
        " VALUES (?,?,?,?,?,?)",
        (uid(), now_iso(), emotion, json.dumps(emotions)[:2000], stress,
         json.dumps(details)[:6000]))
    db().commit()
    return jsonify(id=cur.lastrowid), 201


@app.get("/api/measurements")
@jwt_required()
def list_measurements():
    limit = max(1, min(200, request.args.get("limit", 60, type=int)))
    rows = db().execute("SELECT * FROM measurements WHERE user_id=? ORDER BY id DESC LIMIT ?",
                        (uid(), limit)).fetchall()
    return jsonify([row_to_item(r) for r in rows])


@app.delete("/api/measurements")
@jwt_required()
def clear_measurements():
    db().execute("DELETE FROM measurements WHERE user_id=?", (uid(),))
    db().commit()
    return jsonify(ok=True)


@app.post("/api/measurements/<int:mid>/insight")
@jwt_required()
def insight(mid):
    d = db()
    row = d.execute("SELECT * FROM measurements WHERE id=? AND user_id=?",
                    (mid, uid())).fetchone()
    if not row:
        return jsonify(error="not_found"), 404
    data = request.get_json(silent=True) or {}
    lang = data.get("lang") if data.get("lang") in LANGS else "ru"
    since = (datetime.datetime.now(datetime.timezone.utc)
             - datetime.timedelta(days=30)).isoformat()
    base = d.execute("SELECT AVG(stress) a, COUNT(*) c FROM measurements "
                     "WHERE user_id=? AND id<>? AND created_at>=?",
                     (uid(), mid, since)).fetchone()
    payload = {
        "dominant_emotion": row["emotion"],
        "emotion_probabilities": json.loads(row["emotions_json"]),
        "stress_0_100": row["stress"],
        "details": json.loads(row["details_json"] or "{}"),
        "personal_average_stress_30d": round(base["a"]) if base["c"] else None,
        "previous_analyses_30d": base["c"],
    }
    prompt = (
        "Below is JSON from an automatic analysis of a person's facial expression "
        "(emotion probabilities and facial metrics with tags low/norm/high) and optionally "
        "voice. Write a short personalized analysis in " + LANGS[lang] + " (6-9 sentences, "
        "plain text, no markdown): 1) what the expression suggests, noting that it is only an "
        "estimate, not a diagnosis; 2) how the stress level compares with the person's own "
        "average if available; 3) three concrete practical steps for the next hour. "
        "Be kind and non-alarming.\n\n" + json.dumps(payload, ensure_ascii=False))
    text = ask_ai([{"role": "system", "content": system_prompt(lang)},
                   {"role": "user", "content": prompt}], max_tokens=700)
    if text:
        d.execute("UPDATE measurements SET insight=? WHERE id=?", (text, mid))
        d.commit()
    return jsonify(insight=text, music=music_for(row["emotion"], row["stress"]))

@app.get("/api/chat/history")
@jwt_required()
def chat_history():
    rows = db().execute("SELECT role,content FROM chats WHERE user_id=? "
                        "ORDER BY id DESC LIMIT 50", (uid(),)).fetchall()
    return jsonify([dict(r) for r in rows][::-1])


@app.delete("/api/chat/history")
@jwt_required()
def chat_clear():
    db().execute("DELETE FROM chats WHERE user_id=?", (uid(),))
    db().commit()
    return jsonify(ok=True)


@app.post("/api/chat")
@jwt_required()
def chat():
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()[:2000]
    lang = data.get("lang") if data.get("lang") in LANGS else "ru"
    if not msg:
        return jsonify(error="fill_fields"), 400
    d = db()
    d.execute("INSERT INTO chats(user_id,role,content,created_at) VALUES (?,?,?,?)",
              (uid(), "user", msg, now_iso()))
    if CRISIS_RE.search(msg):
        reply = CRISIS[lang]
    else:
        hist = d.execute("SELECT role,content FROM chats WHERE user_id=? "
                         "ORDER BY id DESC LIMIT 12", (uid(),)).fetchall()[::-1]
        last = d.execute("SELECT emotion,stress FROM measurements WHERE user_id=? "
                         "ORDER BY id DESC LIMIT 1", (uid(),)).fetchone()
        ctx = ""
        if last:
            ctx = ("Latest automatic analysis of the user: emotion=%s, stress=%d/100 "
                   "(rough estimates, not a diagnosis). Use it gently if relevant."
                   % (last["emotion"], last["stress"]))
        messages = [{"role": "system", "content": system_prompt(lang, ctx)}]
        messages += [{"role": r["role"], "content": r["content"]} for r in hist]
        reply = ask_ai(messages)
        if not reply:
            d.commit()
            return jsonify(error="ai_unavailable"), 503
    d.execute("INSERT INTO chats(user_id,role,content,created_at) VALUES (?,?,?,?)",
              (uid(), "assistant", reply, now_iso()))
    d.commit()
    return jsonify(reply=reply)


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 5000)),
            debug=os.getenv("FLASK_DEBUG") == "1")