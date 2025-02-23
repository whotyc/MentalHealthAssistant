import cv2
from fer import FER
import librosa
import numpy as np
import sqlite3
import sounddevice as sd
from flask import Flask, render_template_string, request, session, redirect, url_for, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import asyncio
import aiohttp
import threading
import torch
import torch.nn as nn
from transformers import pipeline
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import matplotlib.pyplot as plt
from io import BytesIO
import base64

detector = FER()
cap = cv2.VideoCapture(0)
app = Flask(__name__)
app.secret_key = "supersecretkey"
jwt = JWTManager(app)
app.config['JWT_SECRET_KEY'] = 'jwt-secret-string'

conn = sqlite3.connect("mental_health.db", check_same_thread=False)
conn.execute('''CREATE TABLE IF NOT EXISTS users 
                (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
conn.execute('''CREATE TABLE IF NOT EXISTS emotions 
                (id INTEGER PRIMARY KEY, emotion TEXT, timestamp TEXT, stress_level REAL, user_id TEXT)''')
conn.execute('''CREATE TABLE IF NOT EXISTS chats 
                (id INTEGER PRIMARY KEY, user_id TEXT, message TEXT, response TEXT, timestamp TEXT)''')
conn.commit()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="YOUR_SPOTIFY_CLIENT_ID",
    client_secret="YOUR_SPOTIFY_CLIENT_SECRET"
))

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.CrossEntropyLoss()

psychologist = pipeline("text-generation", model="distilgpt2")

recommendations = {
    "angry": {"text": "Глубоко подышите.", "spotify": "relaxing piano"},
    "sad": {"text": "Поговорите с другом.", "spotify": "uplifting pop"},
    "happy": {"text": "Сохраняйте позитив!", "spotify": "happy vibes"},
    "stressed": {"text": "Попробуйте медитацию.", "spotify": "calm meditation"}
}

# Анализ голоса
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

# Сохранение
def save_emotion(emotion, stress_level, user_id):
    from datetime import datetime
    conn.execute("INSERT INTO emotions (emotion, timestamp, stress_level, user_id) VALUES (?, ?, ?, ?)", 
                 (emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), stress_level, user_id))
    conn.commit()

def save_chat(user_id, message, response):
    from datetime import datetime
    conn.execute("INSERT INTO chats (user_id, message, response, timestamp) VALUES (?, ?, ?, ?)", 
                 (user_id, message, response, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

def plot_stress(user_id):
    cursor = conn.execute("SELECT stress_level, timestamp FROM emotions WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
    data = cursor.fetchall()
    stress_levels = [row[0] for row in data]
    times = [row[1][-8:] for row in data]  # Только время
    plt.figure(figsize=(6, 4))
    plt.plot(times, stress_levels, marker='o', color='b')
    plt.title("Уровень стресса")
    plt.xticks(rotation=45)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_str

emotion_history = []
def video_processing(user_id):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        emotions = detector.detect_emotions(frame)
        stress_level = analyze_voice()
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
                optimizer.zero_grad()
                outputs = model(torch.tensor(emotion_history[-2][0], dtype=torch.float32).unsqueeze(0))
                target = torch.tensor([["angry", "sad", "happy", "stressed"].index(emotion_history[-2][1])], dtype=torch.long)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

            rec = recommendations.get(pred_emotion, {"text": "Отдохните.", "spotify": "calm"})
            track = sp.search(q=rec["spotify"], type="playlist", limit=1)["playlists"]["items"][0]["external_urls"]["spotify"]
            cv2.putText(frame, f"Emotion: {pred_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Stress: {stress_level:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, rec["text"], (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Spotify: {track}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            save_emotion(pred_emotion, stress_level, user_id)

        cv2.imshow("Mental Health Assistant", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Асинхронные функции
async def async_psychologist(message):
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, lambda: psychologist(message, max_length=100)[0]["generated_text"])
    return response

# Роутинг
@app.route('/login', methods=['GET', 'POST'])
async def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = conn.execute("SELECT id, username FROM users WHERE username=? AND password=?", (username, password))
        user_data = cursor.fetchone()
        if user_data:
            access_token = create_access_token(identity=user_data[0])
            session['username'] = user_data[1]
            return jsonify({"access_token": access_token, "redirect": url_for('home')})
        return "Неверный логин или пароль", 401
    return render_template_string("""
        <style>
            body {font-family: Arial; background: #f0f0f0; text-align: center; padding: 50px;}
            input {padding: 10px; margin: 5px; width: 200px;}
            button {padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer;}
            button:hover {background: #45a049;}
            a {color: #4CAF50; text-decoration: none;}
        </style>
        <h1>Вход</h1>
        <form method="post">
            <input type="text" name="username" placeholder="Логин"><br>
            <input type="password" name="password" placeholder="Пароль"><br>
            <button type="submit">Войти</button>
        </form>
        <a href="/register">Регистрация</a>
    """)

@app.route('/register', methods=['GET', 'POST'])
async def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Пользователь уже существует", 400
    return render_template_string("""
        <style>
            body {font-family: Arial; background: #f0f0f0; text-align: center; padding: 50px;}
            input {padding: 10px; margin: 5px; width: 200px;}
            button {padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer;}
            button:hover {background: #45a049;}
        </style>
        <h1>Регистрация</h1>
        <form method="post">
            <input type="text" name="username" placeholder="Логин"><br>
            <input type="password" name="password" placeholder="Пароль"><br>
            <button type="submit">Зарегистрироваться</button>
        </form>
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

async def history(user_id):
    cursor = conn.execute("SELECT emotion, timestamp, stress_level FROM emotions WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
    history_data = cursor.fetchall()
    stress_plot = plot_stress(user_id)
    return render_template_string("""
        <style>
            body {font-family: Arial; background: #f8f8f8; padding: 20px;}
            h1 {color: #333;}
            .nav {margin-bottom: 20px;}
            button {padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; margin: 5px;}
            button:hover {background: #45a049;}
            table {width: 100%; border-collapse: collapse; margin-top: 20px;}
            th, td {padding: 10px; border: 1px solid #ddd; text-align: left;}
            th {background: #4CAF50; color: white;}
            img {max-width: 100%;}
        </style>
        <h1>Персональный помощник</h1>
        <p>Пользователь: {{username}} <a href="/logout">Выйти</a></p>
        <div class="nav">
            <form method="post" style="display:inline;">
                <input type="hidden" name="tab" value="history">
                <button type="submit">История</button>
            </form>
            <form method="post" style="display:inline;">
                <input type="hidden" name="tab" value="chat">
                <button type="submit">Чат с ИИ</button>
            </form>
        </div>
        <h2>История состояния</h2>
        <img src="data:image/png;base64,{{stress_plot}}" alt="График стресса">
        <table>
            <tr><th>Эмоция</th><th>Время</th><th>Уровень стресса</th></tr>
            {% for row in history %}
                <tr><td>{{row[0]}}</td><td>{{row[1]}}</td><td>{{row[2]}}</td></tr>
            {% endfor %}
        </table>
    """, history=history_data, stress_plot=stress_plot, username=session.get('username', 'User'))

async def chat(user_id):
    if request.method == 'POST' and 'message' in request.form:
        message = request.form['message']
        response = await async_psychologist(f"Ты психолог. Помоги мне: {message}")
        save_chat(user_id, message, response)
    
    cursor = conn.execute("SELECT message, response, timestamp FROM chats WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
    chat_data = cursor.fetchall()
    return render_template_string("""
        <style>
            body {font-family: Arial; background: #f8f8f8; padding: 20px;}
            h1 {color: #333;}
            .nav {margin-bottom: 20px;}
            button {padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; margin: 5px;}
            button:hover {background: #45a049;}
            textarea {width: 100%; height: 100px; margin-top: 10px;}
            .chat {margin-top: 20px; border: 1px solid #ddd; padding: 10px;}
        </style>
        <h1>Персональный помощник</h1>
        <p>Пользователь: {{username}} <a href="/logout">Выйти</a></p>
        <div class="nav">
            <form method="post" style="display:inline;">
                <input type="hidden" name="tab" value="history">
                <button type="submit">История</button>
            </form>
            <form method="post" style="display:inline;">
                <input type="hidden" name="tab" value="chat">
                <button type="submit">Чат с ИИ</button>
            </form>
        </div>
        <h2>Чат с ИИ-психологом</h2>
        <form method="post">
            <input type="hidden" name="tab" value="chat">
            <textarea name="message" placeholder="Напишите свои мысли..."></textarea><br>
            <button type="submit">Отправить</button>
        </form>
        <div class="chat">
            <h3>История чата</h3>
            {% for row in chat %}
                <p><b>Вы ({row[2]}):</b> {{row[0]}}</p>
                <p><b>ИИ:</b> {{row[1]}}</p>
            {% endfor %}
        </div>
    """, chat=chat_data, username=session.get('username', 'User'))

@app.route('/logout')
@jwt_required()
async def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/api/emotions', methods=['GET'])
@jwt_required()
async def api_emotions():
    user_id = get_jwt_identity()
    cursor = conn.execute("SELECT emotion, stress_level, timestamp FROM emotions WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
    return jsonify({"emotions": [{"emotion": row[0], "stress": row[1], "time": row[2]} for row in cursor.fetchall()]})

@app.route('/api/chat', methods=['POST'])
@jwt_required()
async def api_chat():
    user_id = get_jwt_identity()
    message = request.json['message']
    response = await async_psychologist(f"Ты психолог. Помоги мне: {message}")
    save_chat(user_id, message, response)
    return jsonify({"response": response})

if __name__ == "__main__":
    user_id = "1"  
    video_thread = threading.Thread(target=video_processing, args=(user_id,))
    video_thread.start()
    from asyncio import run
    run(app.run(debug=True, use_reloader=False))
    cap.release()
    cv2.destroyAllWindows()
    conn.close()
