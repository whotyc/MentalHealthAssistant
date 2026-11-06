import datetime
import json
import os
import re
import secrets
import sqlite3

from flask import Flask, g, jsonify, render_template, request
from flask_jwt_extended import (JWTManager, create_access_token, get_jwt_identity,
                                jwt_required)
from werkzeug.security import check_password_hash, generate_password_hash

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "mental_health.db")
EMOTIONS = {"happy", "neutral", "sad", "angry", "fearful", "disgusted", "surprised"}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def load_secret():
    """Ключ из переменной окружения или из файла .secret_key (создаётся сам)."""
    env = os.getenv("JWT_SECRET_KEY")
    if env:
        return env
    path = os.path.join(BASE, ".secret_key")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    key = secrets.token_hex(32)
    with open(path, "w") as f:
        f.write(key)
    return key


app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = load_secret()
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = datetime.timedelta(days=7)
jwt = JWTManager(app)


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
        CREATE INDEX IF NOT EXISTS idx_m_user ON measurements(user_id, id);
        """)


init_db()


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def uid():
    return int(get_jwt_identity())


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
    token = create_access_token(identity=str(cur.lastrowid))
    return jsonify(token=token, user={"id": cur.lastrowid, "name": name, "email": email}), 201


@app.post("/api/login")
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    row = db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not row or not check_password_hash(row["password_hash"], data.get("password") or ""):
        return jsonify(error="invalid_credentials"), 401
    token = create_access_token(identity=str(row["id"]))
    return jsonify(token=token, user={"id": row["id"], "name": row["name"], "email": row["email"]})


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
    insight = data.get("insight")
    insight = insight[:3000] if isinstance(insight, str) else None
    if emotion not in EMOTIONS or not isinstance(stress, (int, float)) \
            or not isinstance(emotions, dict):
        return jsonify(error="fill_fields"), 400
    stress = int(max(0, min(100, stress)))
    cur = db().execute(
        "INSERT INTO measurements(user_id,created_at,emotion,emotions_json,stress,details_json,insight)"
        " VALUES (?,?,?,?,?,?,?)",
        (uid(), now_iso(), emotion, json.dumps(emotions)[:2000], stress,
         json.dumps(details)[:6000], insight))
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


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 5000)),
            debug=os.getenv("FLASK_DEBUG") == "1")