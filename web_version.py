import cv2
import asyncio
import concurrent.futures
from fer import FER
import librosa
import numpy as np
import sqlite3
import sounddevice as sd
from flask import Flask, render_template_string, request, session, redirect, url_for, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_socketio import SocketIO, emit
from flask_babel import Babel, gettext, lazy_gettext
import threading
import torch
import torch.nn as nn
from transformers import pipeline
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import speech_recognition as sr
import pyttsx3
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import bcrypt
import os
import pkg_resources
import sys

# Проверка и обновление версии flask-babel
try:
    flask_babel_version = pkg_resources.get_distribution("flask-babel").version
    print(f"Установленная версия flask-babel: {flask_babel_version}")
    if pkg_resources.parse_version(flask_babel_version) < pkg_resources.parse_version("4.0.0"):
        print("Обновление flask-babel до версии >= 4.0.0...")
        sys.exit("Пожалуйста, выполните 'pip install --upgrade flask-babel' и перезапустите скрипт.")
except pkg_resources.DistributionNotFound:
    print("Библиотека flask-babel не установлена. Установите её с помощью 'pip install flask-babel'.")
    sys.exit(1)

# Инициализация
cascade_path = os.path.join(os.path.dirname(__file__), '.venv', 'Lib', 'site-packages', 'cv2', 'data', 'haarcascade_frontalface_default.xml')
if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Haarcascade file not found at {cascade_path}. Please download it from "
                            "https://github.com/opencv/opencv/tree/master/data/haarcascades and place it "
                            "in the cv2/data directory or specify the correct path.")
detector = FER(cascade_file=cascade_path)
cap = cv2.VideoCapture(0)

# Создание приложения Flask
app = Flask(__name__)
app.secret_key = "supersecretkey"
jwt = JWTManager(app)
app.config['JWT_SECRET_KEY'] = 'jwt-secret-string'
app.config['BABEL_DEFAULT_LOCALE'] = 'ru'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = './translations'

# Инициализация Babel с использованием init_app
babel = Babel()
babel.init_app(app)

# Проверка доступности localeselector
if not hasattr(babel, 'localeselector'):
    raise AttributeError(f"Метод localeselector отсутствует в версии flask-babel {flask_babel_version}. "
                         "Убедитесь, что установлена версия >= 4.0.0. Используйте 'pip install --upgrade flask-babel'.")

socketio = SocketIO(app, cors_allowed_origins="*")
scheduler = BackgroundScheduler()

# Переводы
LANGUAGES = ['ru', 'en', 'es', 'fr', 'de', 'zh']

# База данных
conn = sqlite3.connect("mental_health.db", check_same_thread=False)
cursor = conn.cursor()
cursor.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS emotions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        emotion TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        stress_level REAL NOT NULL,
        user_id INTEGER NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        message TEXT NOT NULL,
        response TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS diaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        mood TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
""")
conn.commit()

# Spotify
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="YOUR_SPOTIFY_CLIENT_ID",
    client_secret="YOUR_SPOTIFY_CLIENT_SECRET"
))

# Модель
class EmotionPredictor(nn.Module):
    def __init__(self):
        super(EmotionPredictor, self).__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

model = EmotionPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
psychologist = pipeline("text-generation", model="distilgpt2")

# Рекомендации
recommendations = {
    "angry": {"text": lazy_gettext("Take deep breaths."), "spotify": "relaxing piano"},
    "sad": {"text": lazy_gettext("Talk to a friend."), "spotify": "uplifting pop"},
    "happy": {"text": lazy_gettext("Keep the positivity!"), "spotify": "happy vibes"},
    "stressed": {"text": lazy_gettext("Try meditation."), "spotify": "calm meditation"}
}

# Голосовые функции
recognizer = sr.Recognizer()
engine = pyttsx3.init()

async def analyze_voice_async():
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, analyze_voice)

def analyze_voice():
    duration = 2
    fs = 44100
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio = recording.flatten()
    mfcc = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=fs)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=fs)
    return min(abs(np.mean(mfcc) + np.mean(chroma) + np.mean(spectral_contrast)) / 200, 1.0)

async def save_emotion_async(emotion, stress_level, user_id):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: save_emotion(emotion, stress_level, user_id))

def save_emotion(emotion, stress_level, user_id):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO emotions (emotion, timestamp, stress_level, user_id) VALUES (?, ?, ?, ?)", 
                   (emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), stress_level, user_id))
    conn.commit()

async def save_chat_async(user_id, message, response):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: save_chat(user_id, message, response))

def save_chat(user_id, message, response):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (user_id, message, response, timestamp) VALUES (?, ?, ?, ?)", 
                   (user_id, message, response, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

async def save_diary_async(user_id, content, mood):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: save_diary(user_id, content, mood))

def save_diary(user_id, content, mood):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO diaries (user_id, content, timestamp, mood) VALUES (?, ?, ?, ?)", 
                   (user_id, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mood))
    conn.commit()

def plot_stress(user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT stress_level, timestamp FROM emotions WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
    data = cursor.fetchall()
    stress_levels = [row[0] for row in data]
    times = [row[1][-8:] for row in data]
    plt.figure(figsize=(6, 4))
    plt.plot(times, stress_levels, marker='o', color='b')
    plt.title(gettext("Stress Level"))
    plt.xticks(rotation=45)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_str

emotion_history = []
async def video_processing_async(user_id):
    while True:
        ret, frame = await asyncio.to_thread(lambda: cap.read())
        if not ret:
            break
        emotions = await asyncio.to_thread(lambda: detector.detect_emotions(frame))
        stress_level = await analyze_voice_async()
        if emotions:
            emotion_dict = emotions[0]["emotions"]
            top_emotion = max(emotion_dict, key=emotion_dict.get)
            emotion_data = list(emotion_dict.values()) + [stress_level]
            emotion_history.append((emotion_data, top_emotion))

            with torch.no_grad():
                pred = model(torch.tensor(emotion_data, dtype=torch.float32).unsqueeze(0))
                pred_emotion = ["angry", "sad", "happy", "stressed"][torch.argmax(pred).item()]
            if stress_level > 0.7:
                pred_emotion = "stressed"

            if len(emotion_history) > 10:
                await asyncio.to_thread(lambda: optimizer.zero_grad())
                outputs = model(torch.tensor(emotion_history[-2][0], dtype=torch.float32).unsqueeze(0))
                target = torch.tensor([["angry", "sad", "happy", "stressed"].index(emotion_history[-2][1])], dtype=torch.long)
                loss = criterion(outputs, target)
                await asyncio.to_thread(lambda: loss.backward())
                await asyncio.to_thread(lambda: optimizer.step())

            rec = recommendations.get(pred_emotion, {"text": lazy_gettext("Take a break."), "spotify": "calm"})
            track = sp.search(q=rec["spotify"], type="playlist", limit=1)["playlists"]["items"][0]["external_urls"]["spotify"]
            await asyncio.to_thread(lambda: cv2.putText(frame, f"{gettext('Emotion')}: {pred_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2))
            await asyncio.to_thread(lambda: cv2.putText(frame, f"{gettext('Stress')}: {stress_level:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2))
            await asyncio.to_thread(lambda: cv2.putText(frame, rec["text"], (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2))
            await asyncio.to_thread(lambda: cv2.putText(frame, f"{gettext('Spotify')}: {track}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2))

            await save_emotion_async(pred_emotion, stress_level, user_id)
            socketio.emit('update_emotion', {'emotion': pred_emotion, 'stress': stress_level}, namespace='/notifications')

        await asyncio.to_thread(lambda: cv2.imshow("Mental Health Assistant", frame))
        if await asyncio.to_thread(lambda: cv2.waitKey(1) & 0xFF == ord('q')):
            break

async def recognize_speech_async():
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, recognize_speech)

def recognize_speech():
    with sr.Microphone() as source:
        print(gettext("Listening..."))
        audio = recognizer.listen(source, timeout=5)
    try:
        text = recognizer.recognize_google(audio, language=f"{babel.locale.language}-RU")
        return text
    except sr.UnknownValueError:
        return gettext("Could not recognize speech.")
    except sr.RequestError:
        return gettext("Service recognition error.")

async def text_to_speech_async(text):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: text_to_speech(text))

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

async def check_stress_and_notify_async(user_id):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: check_stress_and_notify(user_id))

def check_stress_and_notify(user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT stress_level FROM emotions WHERE user_id=? ORDER BY timestamp DESC LIMIT 1", (user_id,))
    stress = cursor.fetchone()
    if stress and stress[0] > 0.7:
        socketio.emit('notification', {'message': gettext('Your stress level is high. Try meditation!')}, namespace='/notifications')

# Хеширование пароля
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Роутинг
@app.route('/login', methods=['GET', 'POST'])
async def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash FROM users WHERE username=?", (username,))
        user_data = cursor.fetchone()
        if user_data and check_password(password, user_data[2]):
            access_token = create_access_token(identity=user_data[0])
            session['username'] = user_data[1]
            return jsonify({"access_token": access_token, "redirect": url_for('home')})
        return gettext("Invalid login or password"), 401
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="{{g.lang or 'ru'}}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{gettext('Login — MentalHealthAssistant')}}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {font-family: Arial; background: #f8f8f8; min-height: 100vh; display: flex; justify-content: center; align-items: center;}
                .login-form {background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); width: 100%; max-width: 400px;}
                .form-group {margin-bottom: 15px;}
                .btn-primary {background: #4CAF50; border: none;}
                .btn-primary:hover {background: #45a049;}
                .lang-select {margin-bottom: 15px;}
            </style>
        </head>
        <body>
            <div class="login-form">
                <h1 class="text-center">{{gettext('Login')}}</h1>
                <form method="post">
                    <div class="form-group">
                        <select name="lang" class="form-select lang-select" onchange="window.location.href='/?lang='+this.value">
                            <option value="ru" {% if g.lang == 'ru' %}selected{% endif %}>Русский</option>
                            <option value="en" {% if g.lang == 'en' %}selected{% endif %}>English</option>
                            <option value="es" {% if g.lang == 'es' %}selected{% endif %}>Español</option>
                            <option value="fr" {% if g.lang == 'fr' %}selected{% endif %}>Français</option>
                            <option value="de" {% if g.lang == 'de' %}selected{% endif %}>Deutsch</option>
                            <option value="zh" {% if g.lang == 'zh' %}selected{% endif %}>中文</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <input type="text" name="username" class="form-control" placeholder="{{gettext('Username')}}" required>
                    </div>
                    <div class="form-group">
                        <input type="password" name="password" class="form-control" placeholder="{{gettext('Password')}}" required>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">{{gettext('Log in')}}</button>
                </form>
                <p class="text-center mt-3"><a href="/register" class="text-decoration-none">{{gettext('Register')}}</a></p>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
    """)

@app.route('/register', methods=['GET', 'POST'])
async def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = conn.cursor()
        try:
            password_hash = hash_password(password)
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return gettext("User already exists"), 400
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="{{g.lang or 'ru'}}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{gettext('Register — MentalHealthAssistant')}}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {font-family: Arial; background: #f8f8f8; min-height: 100vh; display: flex; justify-content: center; align-items: center;}
                .login-form {background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); width: 100%; max-width: 400px;}
                .form-group {margin-bottom: 15px;}
                .btn-primary {background: #4CAF50; border: none;}
                .btn-primary:hover {background: #45a049;}
                .lang-select {margin-bottom: 15px;}
            </style>
        </head>
        <body>
            <div class="login-form">
                <h1 class="text-center">{{gettext('Register')}}</h1>
                <form method="post">
                    <div class="form-group">
                        <select name="lang" class="form-select lang-select" onchange="window.location.href='/?lang='+this.value">
                            <option value="ru" {% if g.lang == 'ru' %}selected{% endif %}>Русский</option>
                            <option value="en" {% if g.lang == 'en' %}selected{% endif %}>English</option>
                            <option value="es" {% if g.lang == 'es' %}selected{% endif %}>Español</option>
                            <option value="fr" {% if g.lang == 'fr' %}selected{% endif %}>Français</option>
                            <option value="de" {% if g.lang == 'de' %}selected{% endif %}>Deutsch</option>
                            <option value="zh" {% if g.lang == 'zh' %}selected{% endif %}>中文</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <input type="text" name="username" class="form-control" placeholder="{{gettext('Username')}}" required>
                    </div>
                    <div class="form-group">
                        <input type="password" name="password" class="form-control" placeholder="{{gettext('Password')}}" required>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">{{gettext('Register')}}</button>
                </form>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
    """)

@app.route('/', methods=['GET', 'POST'])
@jwt_required()
async def home():
    user_id = get_jwt_identity()
    tab = request.form.get('tab', 'history')
    
    if tab == 'history':
        return await history(user_id)
    elif tab == 'chat':
        return await chat(user_id)
    elif tab == 'diary':
        return await diary(user_id)

@babel.localeselector
def get_locale():
    return request.args.get('lang', 'ru')

async def history(user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT emotion, timestamp, stress_level FROM emotions WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
    history_data = cursor.fetchall()
    stress_plot = plot_stress(user_id)
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="{{g.lang or 'ru'}}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{gettext('History — MentalHealthAssistant')}}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {font-family: Arial; background: #f8f8f8; padding: 20px;}
                .nav {margin-bottom: 20px;}
                .btn-primary {background: #4CAF50; border: none; margin: 5px;}
                .btn-primary:hover {background: #45a049;}
                table {width: 100%; border-collapse: collapse; margin-top: 20px;}
                th, td {padding: 10px; border: 1px solid #ddd; text-align: left;}
                th {background: #4CAF50; color: white;}
                img {max-width: 100%; margin-top: 20px;}
                .lang-select {margin-bottom: 15px;}
                @media (max-width: 768px) {
                    table, img {width: 100%; font-size: 14px;}
                    .nav {flex-direction: column;}
                }
            </style>
        </head>
        <body>
            <h1>{{gettext('Personal Assistant')}}</h1>
            <p>{{gettext('User:')}} {{username}} <a href="/logout" class="btn btn-danger">{{gettext('Logout')}}</a></p>
            <div class="nav d-flex">
                <form method="post" class="me-2">
                    <input type="hidden" name="tab" value="history">
                    <button type="submit" class="btn btn-primary">{{gettext('History')}}</button>
                </form>
                <form method="post" class="me-2">
                    <input type="hidden" name="tab" value="chat">
                    <button type="submit" class="btn btn-primary">{{gettext('Chat with AI')}}</button>
                </form>
                <form method="post">
                    <input type="hidden" name="tab" value="diary">
                    <button type="submit" class="btn btn-primary">{{gettext('Diary')}}</button>
                </form>
                <select name="lang" class="form-select lang-select" onchange="window.location.href='/?lang='+this.value">
                    <option value="ru" {% if g.lang == 'ru' %}selected{% endif %}>Русский</option>
                    <option value="en" {% if g.lang == 'en' %}selected{% endif %}>English</option>
                    <option value="es" {% if g.lang == 'es' %}selected{% endif %}>Español</option>
                    <option value="fr" {% if g.lang == 'fr' %}selected{% endif %}>Français</option>
                    <option value="de" {% if g.lang == 'de' %}selected{% endif %}>Deutsch</option>
                    <option value="zh" {% if g.lang == 'zh' %}selected{% endif %}>中文</option>
                </select>
            </div>
            <h2>{{gettext('History of States')}}</h2>
            <img src="data:image/png;base64,{{stress_plot}}" alt="{{gettext('Stress Chart')}}" class="img-fluid">
            <table>
                <tr><th>{{gettext('Emotion')}}</th><th>{{gettext('Time')}}</th><th>{{gettext('Stress Level')}}</th></tr>
                {% for row in history %}
                    <tr><td>{{row[0]}}</td><td>{{row[1]}}</td><td>{{row[2]}}</td></tr>
                {% endfor %}
            </table>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
            <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
            <script>
                const socket = io.connect('http://localhost:5000', {path: '/socket.io'});
                socket.on('notification', (data) => {
                    alert(data.message);
                });
                socket.on('update_emotion', (data) => {
                    console.log('Emotion updated:', data);
                });
            </script>
        </body>
        </html>
    """, history=history_data, stress_plot=stress_plot, username=session.get('username', 'User'))

async def chat(user_id):
    if request.method == 'POST' and 'message' in request.form:
        message = request.form['message']
        response = await async_psychologist(f"You are a psychologist. Help me: {message}")
        await save_chat_async(user_id, message, response)
    elif request.method == 'POST' and 'voice' in request.form:
        text = await recognize_speech_async()
        if text not in [gettext("Could not recognize speech."), gettext("Service recognition error.")]:
            response = await async_psychologist(f"You are a psychologist. Help me: {text}")
            await save_chat_async(user_id, text, response)
            await text_to_speech_async(response)

    cursor = conn.cursor()
    cursor.execute("SELECT message, response, timestamp FROM chats WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
    chat_data = cursor.fetchall()
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="{{g.lang or 'ru'}}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{gettext('Chat — MentalHealthAssistant')}}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {font-family: Arial; background: #f8f8f8; padding: 20px;}
                .nav {margin-bottom: 20px;}
                .btn-primary {background: #4CAF50; border: none; margin: 5px;}
                .btn-primary:hover {background: #45a049;}
                textarea {width: 100%; height: 100px; margin-top: 10px;}
                .chat {margin-top: 20px; border: 1px solid #ddd; padding: 10px; max-height: 300px; overflow-y: auto;}
                .voice-btn {margin-top: 10px;}
                .lang-select {margin-bottom: 15px;}
                @media (max-width: 768px) {
                    textarea, .chat {width: 100%; font-size: 14px;}
                    .nav {flex-direction: column;}
                }
            </style>
        </head>
        <body>
            <h1>{{gettext('Personal Assistant')}}</h1>
            <p>{{gettext('User:')}} {{username}} <a href="/logout" class="btn btn-danger">{{gettext('Logout')}}</a></p>
            <div class="nav d-flex">
                <form method="post" class="me-2">
                    <input type="hidden" name="tab" value="history">
                    <button type="submit" class="btn btn-primary">{{gettext('History')}}</button>
                </form>
                <form method="post" class="me-2">
                    <input type="hidden" name="tab" value="chat">
                    <button type="submit" class="btn btn-primary">{{gettext('Chat with AI')}}</button>
                </form>
                <form method="post">
                    <input type="hidden" name="tab" value="diary">
                    <button type="submit" class="btn btn-primary">{{gettext('Diary')}}</button>
                </form>
                <select name="lang" class="form-select lang-select" onchange="window.location.href='/?lang='+this.value">
                    <option value="ru" {% if g.lang == 'ru' %}selected{% endif %}>Русский</option>
                    <option value="en" {% if g.lang == 'en' %}selected{% endif %}>English</option>
                    <option value="es" {% if g.lang == 'es' %}selected{% endif %}>Español</option>
                    <option value="fr" {% if g.lang == 'fr' %}selected{% endif %}>Français</option>
                    <option value="de" {% if g.lang == 'de' %}selected{% endif %}>Deutsch</option>
                    <option value="zh" {% if g.lang == 'zh' %}selected{% endif %}>中文</option>
                </select>
            </div>
            <h2>{{gettext('Chat with AI Psychologist')}}</h2>
            <form method="post">
                <input type="hidden" name="tab" value="chat">
                <textarea name="message" placeholder="{{gettext('Write your thoughts...')}}"></textarea><br>
                <button type="submit" class="btn btn-primary">{{gettext('Send')}}</button>
                <button type="submit" name="voice" class="btn btn-secondary voice-btn">{{gettext('Voice Input')}}</button>
            </form>
            <div class="chat">
                <h3>{{gettext('Chat History')}}</h3>
                {% for row in chat %}
                    <p><b>{{gettext('You (')}}{{row[2]}}):</b> {{row[0]}}</p>
                    <p><b>{{gettext('AI:')}}</b> {{row[1]}}</p>
                {% endfor %}
            </div>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
            <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
            <script>
                const socket = io.connect('http://localhost:5000', {path: '/socket.io'});
                socket.on('notification', (data) => {
                    alert(data.message);
                });
                document.querySelector('.voice-btn').addEventListener('click', async () => {
                    const recognition = new webkitSpeechRecognition() || new SpeechRecognition();
                    recognition.lang = '{{g.lang or "ru-RU"}}'.split('-')[0] + '-RU';
                    recognition.start();
                    recognition.onresult = async (event) => {
                        const text = event.results[0][0].transcript;
                        const formData = new FormData();
                        formData.append('voice', text);
                        fetch('/', {
                            method: 'POST',
                            body: formData
                        }).then(response => response.text()).then(data => {
                            alert('AI responded: ' + data);
                        });
                    };
                });
            </script>
        </body>
        </html>
    """, chat=chat_data, username=session.get('username', 'User'))

async def diary(user_id):
    if request.method == 'POST' and 'content' in request.form and 'mood' in request.form:
        content = request.form['content']
        mood = request.form['mood']
        await save_diary_async(user_id, content, mood)

    cursor = conn.cursor()
    cursor.execute("SELECT content, timestamp, mood FROM diaries WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
    diary_data = cursor.fetchall()
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="{{g.lang or 'ru'}}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{gettext('Diary — MentalHealthAssistant')}}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {font-family: Arial; background: #f8f8f8; padding: 20px;}
                .nav {margin-bottom: 20px;}
                .btn-primary {background: #4CAF50; border: none; margin: 5px;}
                .btn-primary:hover {background: #45a049;}
                textarea {width: 100%; height: 150px; margin-top: 10px;}
                .diary-entry {margin-top: 20px; border: 1px solid #ddd; padding: 10px; max-height: 300px; overflow-y: auto;}
                .lang-select {margin-bottom: 15px;}
                @media (max-width: 768px) {
                    textarea, .diary-entry {width: 100%; font-size: 14px;}
                    .nav {flex-direction: column;}
                }
            </style>
        </head>
        <body>
            <h1>{{gettext('Personal Assistant')}}</h1>
            <p>{{gettext('User:')}} {{username}} <a href="/logout" class="btn btn-danger">{{gettext('Logout')}}</a></p>
            <div class="nav d-flex">
                <form method="post" class="me-2">
                    <input type="hidden" name="tab" value="history">
                    <button type="submit" class="btn btn-primary">{{gettext('History')}}</button>
                </form>
                <form method="post" class="me-2">
                    <input type="hidden" name="tab" value="chat">
                    <button type="submit" class="btn btn-primary">{{gettext('Chat with AI')}}</button>
                </form>
                <form method="post">
                    <input type="hidden" name="tab" value="diary">
                    <button type="submit" class="btn btn-primary">{{gettext('Diary')}}</button>
                </form>
                <select name="lang" class="form-select lang-select" onchange="window.location.href='/?lang='+this.value">
                    <option value="ru" {% if g.lang == 'ru' %}selected{% endif %}>Русский</option>
                    <option value="en" {% if g.lang == 'en' %}selected{% endif %}>English</option>
                    <option value="es" {% if g.lang == 'es' %}selected{% endif %}>Español</option>
                    <option value="fr" {% if g.lang == 'fr' %}selected{% endif %}>Français</option>
                    <option value="de" {% if g.lang == 'de' %}selected{% endif %}>Deutsch</option>
                    <option value="zh" {% if g.lang == 'zh' %}selected{% endif %}>中文</option>
                </select>
            </div>
            <h2>{{gettext('Diary')}}</h2>
            <form method="post">
                <input type="hidden" name="tab" value="diary">
                <textarea name="content" placeholder="{{gettext('Write your thoughts...')}}"></textarea><br>
                <select name="mood" class="form-select" required>
                    <option value="happy">{{gettext('Happy')}}</option>
                    <option value="sad">{{gettext('Sad')}}</option>
                    <option value="angry">{{gettext('Angry')}}</option>
                    <option value="stressed">{{gettext('Stressed')}}</option>
                </select>
                <button type="submit" class="btn btn-primary mt-2">{{gettext('Save')}}</button>
            </form>
            <div class="diary-entry">
                <h3>{{gettext('Recent Entries')}}</h3>
                {% for row in diary %}
                    <p><b>{{row[1]}} ({{gettext('Mood:')}} {{row[2]}}):</b> {{row[0]}}</p>
                {% endfor %}
            </div>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
            <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
            <script>
                const socket = io.connect('http://localhost:5000', {path: '/socket.io'});
                socket.on('notification', (data) => {
                    alert(data.message);
                });
            </script>
        </body>
        </html>
    """, diary=diary_data, username=session.get('username', 'User'))

@app.route('/logout')
@jwt_required()
async def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# API
@app.route('/api/emotions', methods=['GET'])
@jwt_required()
async def api_emotions():
    user_id = get_jwt_identity()
    cursor = conn.cursor()
    cursor.execute("SELECT emotion, stress_level, timestamp FROM emotions WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
    return jsonify({"emotions": [{"emotion": row[0], "stress": row[1], "time": row[2]} for row in cursor.fetchall()]})

@app.route('/api/chat', methods=['POST'])
@jwt_required()
async def api_chat():
    user_id = get_jwt_identity()
    if 'message' in request.json:
        message = request.json['message']
        response = await async_psychologist(f"You are a psychologist. Help me: {message}")
        await save_chat_async(user_id, message, response)
        return jsonify({"response": response})
    elif 'voice' in request.json:
        text = await recognize_speech_async()
        if text not in [gettext("Could not recognize speech."), gettext("Service recognition error.")]:
            response = await async_psychologist(f"You are a psychologist. Help me: {text}")
            await save_chat_async(user_id, text, response)
            await text_to_speech_async(response)
            return jsonify({"response": response})
        return jsonify({"response": gettext("Voice input error")}), 400

@app.route('/api/diary', methods=['POST'])
@jwt_required()
async def api_diary():
    user_id = get_jwt_identity()
    content = request.json.get('content', '')
    mood = request.json.get('mood', '')
    if content and mood:
        await save_diary_async(user_id, content, mood)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": gettext("Missing data")}), 400

@socketio.on('connect', namespace='/notifications')
@jwt_required()
def handle_connect():
    user_id = get_jwt_identity()
    emit('welcome', {'message': f'Connected user {user_id}'})

async def start_video_processing(user_id):
    await video_processing_async(user_id)

if __name__ == "__main__":
    user_id = "1"  # Заменится после входа
    video_thread = threading.Thread(target=lambda: asyncio.run(start_video_processing(str(user_id))))
    video_thread.start()
    scheduler.add_job(lambda: asyncio.run(check_stress_and_notify_async(str(user_id))), 'interval', minutes=30)
    scheduler.start()
    socketio.run(app, debug=True, use_reloader=False)
    cap.release()
    cv2.destroyAllWindows()
    conn.close()