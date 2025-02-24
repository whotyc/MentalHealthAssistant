import cv2
from fer import FER
import librosa
import numpy as np
import sqlite3
import sounddevice as sd
import asyncio
from flask import Flask, render_template_string, request, session, redirect, url_for, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_socketio import SocketIO, emit
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

detector = FER()
cap = cv2.VideoCapture(0)
app = Flask(__name__)
app.secret_key = "supersecretkey"
jwt = JWTManager(app)
app.config['JWT_SECRET_KEY'] = 'jwt-secret-string'
socketio = SocketIO(app, cors_allowed_origins="*")
scheduler = BackgroundScheduler()

conn = sqlite3.connect("mental_health.db", check_same_thread=False)

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
psychologist = pipeline("text-generation", model="distilgpt2")

recommendations = {
    "angry": {"text": "Глубоко подышите.", "spotify": "relaxing piano"},
    "sad": {"text": "Поговорите с другом.", "spotify": "uplifting pop"},
    "happy": {"text": "Сохраняйте позитив!", "spotify": "happy vibes"},
    "stressed": {"text": "Попробуйте медитацию.", "spotify": "calm meditation"}
}

recognizer = sr.Recognizer()
engine = pyttsx3.init()

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

def save_emotion(emotion, stress_level, user_id):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO emotions (emotion, timestamp, stress_level, user_id) VALUES (?, ?, ?, ?)", 
                   (emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), stress_level, user_id))
    conn.commit()

def save_chat(user_id, message, response):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (user_id, message, response, timestamp) VALUES (?, ?, ?, ?)", 
                   (user_id, message, response, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

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
            cv2.putText(frame, f"Эмоция: {pred_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Стресс: {stress_level:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, rec["text"], (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Spotify: {track}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            save_emotion(pred_emotion, stress_level, user_id)
            socketio.emit('update_emotion', {'emotion': pred_emotion, 'stress': stress_level}, namespace='/notifications')

        cv2.imshow("Mental Health Assistant", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

async def async_psychologist(message):
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, lambda: psychologist(f"Ты психолог. Помоги мне: {message}", max_length=100)[0]["generated_text"])
    return response

def recognize_speech():
    with sr.Microphone() as source:
        print("Слушаю...")
        audio = recognizer.listen(source, timeout=5)
    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        return text
    except sr.UnknownValueError:
        return "Не удалось распознать речь."
    except sr.RequestError:
        return "Ошибка сервиса распознавания."

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

def check_stress_and_notify(user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT stress_level FROM emotions WHERE user_id=? ORDER BY timestamp DESC LIMIT 1", (user_id,))
    stress = cursor.fetchone()
    if stress and stress[0] > 0.7:
        socketio.emit('notification', {'message': 'Ваш уровень стресса высокий. Попробуйте медитацию!'}, namespace='/notifications')

scheduler.add_job(lambda: check_stress_and_notify(str(get_jwt_identity())), 'interval', minutes=30)
scheduler.start()

@app.route('/login', methods=['GET', 'POST'])
async def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = conn.cursor()
        cursor.execute("SELECT id, username FROM users WHERE username=? AND password=?", (username, password))
        user_data = cursor.fetchone()
        if user_data:
            access_token = create_access_token(identity=user_data[0])
            session['username'] = user_data[1]
            return jsonify({"access_token": access_token, "redirect": url_for('home')})
        return "Неверный логин или пароль", 401
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Вход — MentalHealthAssistant</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {font-family: Arial; background: #f8f8f8; min-height: 100vh; display: flex; justify-content: center; align-items: center;}
                .login-form {background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); width: 100%; max-width: 400px;}
                .form-group {margin-bottom: 15px;}
                .btn-primary {background: #4CAF50; border: none;}
                .btn-primary:hover {background: #45a049;}
            </style>
        </head>
        <body>
            <div class="login-form">
                <h1 class="text-center">Вход</h1>
                <form method="post">
                    <div class="form-group">
                        <input type="text" name="username" class="form-control" placeholder="Логин" required>
                    </div>
                    <div class="form-group">
                        <input type="password" name="password" class="form-control" placeholder="Пароль" required>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Войти</button>
                </form>
                <p class="text-center mt-3"><a href="/register" class="text-decoration-none">Регистрация</a></p>
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
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Пользователь уже существует", 400
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Регистрация — MentalHealthAssistant</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {font-family: Arial; background: #f8f8f8; min-height: 100vh; display: flex; justify-content: center; align-items: center;}
                .login-form {background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); width: 100%; max-width: 400px;}
                .form-group {margin-bottom: 15px;}
                .btn-primary {background: #4CAF50; border: none;}
                .btn-primary:hover {background: #45a049;}
            </style>
        </head>
        <body>
            <div class="login-form">
                <h1 class="text-center">Регистрация</h1>
                <form method="post">
                    <div class="form-group">
                        <input type="text" name="username" class="form-control" placeholder="Логин" required>
                    </div>
                    <div class="form-group">
                        <input type="password" name="password" class="form-control" placeholder="Пароль" required>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Зарегистрироваться</button>
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

async def history(user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT emotion, timestamp, stress_level FROM emotions WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
    history_data = cursor.fetchall()
    stress_plot = plot_stress(user_id)
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>История — MentalHealthAssistant</title>
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
                @media (max-width: 768px) {
                    table, img {width: 100%; font-size: 14px;}
                    .nav {flex-direction: column;}
                }
            </style>
        </head>
        <body>
            <h1>Персональный помощник</h1>
            <p>Пользователь: {{username}} <a href="/logout" class="btn btn-danger">Выйти</a></p>
            <div class="nav d-flex">
                <form method="post" class="me-2">
                    <input type="hidden" name="tab" value="history">
                    <button type="submit" class="btn btn-primary">История</button>
                </form>
                <form method="post" class="me-2">
                    <input type="hidden" name="tab" value="chat">
                    <button type="submit" class="btn btn-primary">Чат с ИИ</button>
                </form>
                <form method="post">
                    <input type="hidden" name="tab" value="diary">
                    <button type="submit" class="btn btn-primary">Дневник</button>
                </form>
            </div>
            <h2>История состояния</h2>
            <img src="data:image/png;base64,{{stress_plot}}" alt="График стресса" class="img-fluid">
            <table>
                <tr><th>Эмоция</th><th>Время</th><th>Уровень стресса</th></tr>
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
                    console.log('Эмоция обновлена:', data);
                });
            </script>
        </body>
        </html>
    """, history=history_data, stress_plot=stress_plot, username=session.get('username', 'User'))

async def chat(user_id):
    if request.method == 'POST' and 'message' in request.form:
        message = request.form['message']
        response = await async_psychologist(f"Ты психолог. Помоги мне: {message}")
        save_chat(user_id, message, response)
    elif request.method == 'POST' and 'voice' in request.form:
        text = recognize_speech()
        if text != "Не удалось распознать речь." and text != "Ошибка сервиса распознавания.":
            response = await async_psychologist(f"Ты психолог. Помоги мне: {text}")
            save_chat(user_id, text, response)
            text_to_speech(response)

    cursor = conn.cursor()
    cursor.execute("SELECT message, response, timestamp FROM chats WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
    chat_data = cursor.fetchall()
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Чат — MentalHealthAssistant</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {font-family: Arial; background: #f8f8f8; padding: 20px;}
                .nav {margin-bottom: 20px;}
                .btn-primary {background: #4CAF50; border: none; margin: 5px;}
                .btn-primary:hover {background: #45a049;}
                textarea {width: 100%; height: 100px; margin-top: 10px;}
                .chat {margin-top: 20px; border: 1px solid #ddd; padding: 10px; max-height: 300px; overflow-y: auto;}
                .voice-btn {margin-top: 10px;}
                @media (max-width: 768px) {
                    textarea, .chat {width: 100%; font-size: 14px;}
                    .nav {flex-direction: column;}
                }
            </style>
        </head>
        <body>
            <h1>Персональный помощник</h1>
            <p>Пользователь: {{username}} <a href="/logout" class="btn btn-danger">Выйти</a></p>
            <div class="nav d-flex">
                <form method="post" class="me-2">
                    <input type="hidden" name="tab" value="history">
                    <button type="submit" class="btn btn-primary">История</button>
                </form>
                <form method="post" class="me-2">
                    <input type="hidden" name="tab" value="chat">
                    <button type="submit" class="btn btn-primary">Чат с ИИ</button>
                </form>
                <form method="post">
                    <input type="hidden" name="tab" value="diary">
                    <button type="submit" class="btn btn-primary">Дневник</button>
                </form>
            </div>
            <h2>Чат с ИИ-психологом</h2>
            <form method="post">
                <input type="hidden" name="tab" value="chat">
                <textarea name="message" placeholder="Напишите свои мысли..."></textarea><br>
                <button type="submit" class="btn btn-primary">Отправить</button>
                <button type="submit" name="voice" class="btn btn-secondary voice-btn">Голосовой ввод</button>
            </form>
            <div class="chat">
                <h3>История чата</h3>
                {% for row in chat %}
                    <p><b>Вы ({{row[2]}}):</b> {{row[0]}}</p>
                    <p><b>ИИ:</b> {{row[1]}}</p>
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
                    recognition.lang = 'ru-RU';
                    recognition.start();
                    recognition.onresult = async (event) => {
                        const text = event.results[0][0].transcript;
                        const formData = new FormData();
                        formData.append('voice', text);
                        fetch('/', {
                            method: 'POST',
                            body: formData
                        }).then(response => response.text()).then(data => {
                            alert('ИИ ответил: ' + data);
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
        save_diary(user_id, content, mood)

    cursor = conn.cursor()
    cursor.execute("SELECT content, timestamp, mood FROM diaries WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (user_id,))
    diary_data = cursor.fetchall()
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Дневник — MentalHealthAssistant</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {font-family: Arial; background: #f8f8f8; padding: 20px;}
                .nav {margin-bottom: 20px;}
                .btn-primary {background: #4CAF50; border: none; margin: 5px;}
                .btn-primary:hover {background: #45a049;}
                textarea {width: 100%; height: 150px; margin-top: 10px;}
                .diary-entry {margin-top: 20px; border: 1px solid #ddd; padding: 10px; max-height: 300px; overflow-y: auto;}
                @media (max-width: 768px) {
                    textarea, .diary-entry {width: 100%; font-size: 14px;}
                    .nav {flex-direction: column;}
                }
            </style>
        </head>
        <body>
            <h1>Персональный помощник</h1>
            <p>Пользователь: {{username}} <a href="/logout" class="btn btn-danger">Выйти</a></p>
            <div class="nav d-flex">
                <form method="post" class="me-2">
                    <input type="hidden" name="tab" value="history">
                    <button type="submit" class="btn btn-primary">История</button>
                </form>
                <form method="post" class="me-2">
                    <input type="hidden" name="tab" value="chat">
                    <button type="submit" class="btn btn-primary">Чат с ИИ</button>
                </form>
                <form method="post">
                    <input type="hidden" name="tab" value="diary">
                    <button type="submit" class="btn btn-primary">Дневник</button>
                </form>
            </div>
            <h2>Дневник</h2>
            <form method="post">
                <input type="hidden" name="tab" value="diary">
                <textarea name="content" placeholder="Запишите свои мысли..."></textarea><br>
                <select name="mood" class="form-select" required>
                    <option value="happy">Счастливый</option>
                    <option value="sad">Грустный</option>
                    <option value="angry">Злой</option>
                    <option value="stressed">Стресс</option>
                </select>
                <button type="submit" class="btn btn-primary mt-2">Сохранить</button>
            </form>
            <div class="diary-entry">
                <h3>Последние записи</h3>
                {% for row in diary %}
                    <p><b>{{row[1]}} (Настроение: {{row[2]}}):</b> {{row[0]}}</p>
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
        response = await async_psychologist(f"Ты психолог. Помоги мне: {message}")
        save_chat(user_id, message, response)
        return jsonify({"response": response})
    elif 'voice' in request.json:
        text = recognize_speech()
        if text != "Не удалось распознать речь." and text != "Ошибка сервиса распознавания.":
            response = await async_psychologist(f"Ты психолог. Помоги мне: {text}")
            save_chat(user_id, text, response)
            return jsonify({"response": response})
        return jsonify({"response": "Ошибка голосового ввода"}), 400

@app.route('/api/diary', methods=['POST'])
@jwt_required()
async def api_diary():
    user_id = get_jwt_identity()
    content = request.json.get('content', '')
    mood = request.json.get('mood', '')
    if content and mood:
        save_diary(user_id, content, mood)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Не хватает данных"}), 400

@socketio.on('connect', namespace='/notifications')
@jwt_required()
def handle_connect():
    user_id = get_jwt_identity()
    emit('welcome', {'message': f'Подключен пользователь {user_id}'})

if __name__ == "__main__":
    user_id = "1"  # Заменится после входа
    video_thread = threading.Thread(target=video_processing, args=(user_id,))
    video_thread.start()
    scheduler.start()
    socketio.run(app, debug=True, use_reloader=False)
    cap.release()
    cv2.destroyAllWindows()
    conn.close()
